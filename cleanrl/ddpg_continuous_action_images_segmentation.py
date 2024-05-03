# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import copy
import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from custom_env import MedicalImageSegmentationEnv
from pathlib import Path
from torchsummary import summary


@dataclass
class Args:
    exp_name: str = "Image_Segmentation-ResNet"
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
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Image_Segmentation-v1"
    """the environment id"""
    total_timesteps: int = int(3e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(5e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = int(25e2)
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space):
    def thunk():
        env = MedicalImageSegmentationEnv(data_path, num_control_points, max_iter, iou_threshold, interval_action_space)
        return env
    return thunk


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """

        :param x: the input tensor containing both mask (channel 0) and ground truth (channel 1) data in shape [batch_size, channels=2, height=110, width=110] ; mask and ground truth are binary images [0, 1]
        :return: the Q value of the input state-action pair in shape [batch_size, 1]
        """
        x = x.float()
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Assuming env.observation_space.shape is (H, W, C)
        input_channels = env.observation_space.shape[2] - 1  # Last dimension is channels AND -1 because i only use mask and image and not the ground truth
        self.feature_extractor = ResNetFeatureExtractor(input_channels)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))

        # Batch normalization layers for each channel
        self.bn_image = nn.BatchNorm2d(1)  # BatchNorm for the image channel
        self.bn_mask = nn.BatchNorm2d(1)   # BatchNorm for the mask channel

        # Action rescaling buffers
        self.register_buffer(
            "action_scale", torch.tensor(((env.action_space.high - env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(((env.action_space.high + env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32)
        )

    def forward(self, x):
        """
        :param x:  the input tensor containing both image (channel 0) and mask (channel 1) data in shape [batch_size, channels=2, height=110, width=110] ; image is grayscale [0,255] and mask is binary [0, 1]
        :return: the action parameters to take with the shape [batch_size, action_dim]
        """
        x = x.float()
        image = x[:, 0:1, :, :]
        mask = x[:, 1:2, :, :]
        image = self.bn_image(image)
        mask = self.bn_mask(mask)

        x = torch.cat([image, mask], dim=1)

        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x)).view(-1, *env.action_space.shape)
        output = (x.view(-1, np.prod(env.action_space.shape)) * self.action_scale) + self.action_bias
        return output


if __name__ == "__main__":
    data_path = Path('..') / 'synthetic_ds' / 'synthetic_dataset.h5'
    num_control_points = 4
    max_iter = 1
    iou_threshold = 0.5
    interval_action_space = 0.125

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__CP{num_control_points}__AS{interval_action_space}__it{max_iter}__{int(time.time())}"
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
    env = make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space)()

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # Assuming the input size for both models is (2, 110, 110)
    input_size = (2, 110, 110)

    print("Actor Summary:")
    summary(actor, input_size)

    print("\nQNetwork (qf1) Summary:")
    summary(qf1, input_size)
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
        if global_step % 100 == 0:
            env.render()

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
                qf1_next_target = qf1_target(temp_qf1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            temp_qf1_a_values = copy.deepcopy(data.observations)
            temp_qf1_a_values = temp_qf1_a_values[:, :, :, [1, 2]].permute(0, 3, 1, 2)
            qf1_a_values = qf1(temp_qf1_a_values).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                temp = copy.deepcopy(data.observations)
                temp = temp[:, :, :, [1, 2]].permute(0, 3, 1, 2)
                actor_loss = -qf1(temp).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

        from ddpg_eval import evaluate

        episodic_returns, obs_env = evaluate(
        model_path,
        make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space),
        eval_episodes=100,
        run_name="eval",
        Model=(Actor, QNetwork),
        device="cpu",
    )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            print(f"eval_episode={idx}, episodic_return={episodic_return}")
            img = obs_env[idx]
            img = img.transpose(2, 0, 1)  # Reshape the image to (C, H, W) format
            writer.add_image(f"charts/obs_eval{idx}", img, global_step)

    env.close()
    writer.close()

    end_time = time.time()
    execution_time = end_time - start_time  # Time in seconds
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = (execution_time % 3600) % 60

    print(f"Execution time: {int(hours)}:{int(minutes)}:{int(seconds)}")
