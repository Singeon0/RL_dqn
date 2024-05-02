# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import copy
import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
from gymnasium.wrappers import ResizeObservation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from custom_env import MedicalImageSegmentationEnv
from pathlib import Path
from torchsummary import summary


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Image_Segmentation-v0"
    """the environment id of the Atari game"""
    total_timesteps: int = int(1e1)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(3e1)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = int(25e0)
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(data_path, num_control_points, max_iter, iou_threshold):
    def thunk():
        env = MedicalImageSegmentationEnv(data_path, num_control_points, max_iter, iou_threshold)
        return env
    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * env.observation_space.shape[0] * env.observation_space.shape[1] + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        # x is now assumed to be the input tensor containing both image and ground truth data in shape [16, 2, 110, 110]
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output for the fully connected layers

        # Flatten the action tensor if not already flat
        action = action.view(action.size(0), -1)

        # Concatenate the flattened feature maps and action
        x = torch.cat([x, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Assuming env.observation_space.shape is (H, W, C)
        input_channels = env.observation_space.shape[2] - 1  # Last dimension is channels AND -1 because i only use mask and image and not the ground truth
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # After convolutions, the spatial dimensions (H, W) remain unchanged due to padding
        self.fc1 = nn.Linear(64 * env.observation_space.shape[0] * env.observation_space.shape[1], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))

        # Action rescaling buffers
        self.register_buffer(
            "action_scale", torch.tensor(((env.action_space.high - env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(((env.action_space.high + env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32)
        )

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Reshape the output for the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x)).view(-1, *env.action_space.shape)
        output = (x.view(-1, np.prod(env.action_space.shape)) * self.action_scale) + self.action_bias
        # print(f"Output shape: {output.shape}")
        return output


if __name__ == "__main__":
    data_path = Path('..') / 'synthetic_ds' / 'synthetic_dataset.h5'
    num_control_points = 4
    max_iter = 4
    iou_threshold = 0.85

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Device: {device}")

    # env setup
    env = make_env(data_path, num_control_points, max_iter, iou_threshold)()

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)


    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Initialize a variable to keep track of the last printed percentage
    last_printed_percentage = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset()
    for global_step in range(args.total_timesteps):
        # Calculate the current percentage of total timesteps completed
        current_percentage = (global_step / args.total_timesteps) * 100

        # If the current percentage is at least 10% more than the last printed percentage, print a message
        if current_percentage - last_printed_percentage >= 10:
            print(f"Script executed at {current_percentage:.0f}%")
            last_printed_percentage = current_percentage

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = np.round(env.action_space.sample(), 2)
        else:
            with torch.no_grad():
                temp = copy.deepcopy(obs)
                temp = torch.from_numpy(temp[:, :, 0:2]).permute(2, 0, 1).unsqueeze(0)
                action = actor(torch.Tensor(temp).to(device))
                action += torch.normal(0, actor.action_scale * args.exploration_noise)
                action = action.cpu().numpy().reshape(env.action_space.shape).clip(env.action_space.low,
                                                                                   env.action_space.high)  # I MODIFY THAT
                action = np.round(action, 2)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, truncated, info = env.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in info:
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if terminated or truncated:
            real_next_obs = info["final_observation"]
            rb.add(obs, real_next_obs, action, reward, terminated, info)  # I MODIFY THAT
            next_obs, _ = env.reset()  # I MODIFY THAT

        # I MODIFY THAT
        else:
            rb.add(obs, real_next_obs, action, reward, terminated, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                temp_actor = copy.deepcopy(data.next_observations)
                temp_actor = temp_actor[:, :, :, 0:2].permute(0, 3, 1, 2)
                next_state_actions = target_actor(temp_actor)
                temp_qf1 = copy.deepcopy(data.next_observations)
                temp_qf1 = temp_qf1[:, :, :, [1, 2]].permute(0, 3, 1, 2)
                qf1_next_target = qf1_target(temp_qf1, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            temp_qf1_a_values = copy.deepcopy(data.observations)
            temp_qf1_a_values = temp_qf1_a_values[:, :, :, [1, 2]].permute(0, 3, 1, 2)
            qf1_a_values = qf1(temp_qf1_a_values, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                temp = copy.deepcopy(data.observations)
                temp = temp[:, :, :, [1, 2]].permute(0, 3, 1, 2)
                actor_loss = -qf1(temp, actor(temp)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:  # TODO i don"t enter in this loop ??
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
        model_path,
        make_env(data_path, num_control_points, max_iter, iou_threshold),
        eval_episodes=10,
        run_name="eval",
        Model=(Actor, QNetwork),
        device="cpu",
    )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            print(f"eval_episode={idx}, episodic_return={episodic_return}")

    env.close()
    writer.close()

    end_time = time.time()
    execution_time = end_time - start_time  # Time in seconds
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = (execution_time % 3600) % 60

    print(f"Execution time: {int(hours)}:{int(minutes)}:{int(seconds)}")
