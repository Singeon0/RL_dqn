# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tyro
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from custom_env import MedicalImageSegmentationEnv

import warnings
warnings.filterwarnings("ignore")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    data_path: Path = Path('..') / 'synthetic_ds' / 'synthetic_dataset.h5'
    """Path to the synthetic dataset"""
    num_control_points: int = 4
    """Number of control points"""
    max_iter: int = 100
    """Maximum number of iterations"""
    iou_threshold: float = 0.8
    """Intersection over Union (IoU) threshold"""
    interval_action_space: float = 0.15
    """Interval of the action space"""
    iou_truncate: float = 0.1
    """IoU truncate value"""
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
    """whether to save model into the `runs_ppo/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Seg_PPO-v0"
    """the id of the environment"""
    total_timesteps: int = 1e5
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 100
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space, iou_truncate):
    def thunk():
        env = MedicalImageSegmentationEnv(data_path, num_control_points, max_iter, iou_threshold, interval_action_space, iou_truncate)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.feature_extractor = ResNetFeatureExtractor(np.array(envs.single_observation_space.shape)[2] - 1)

        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(envs.single_action_space.shape)),
            nn.Tanh()  # Assuming the action space is normalized between -1 and 1
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)  # Permute the dimensions to (batch_size, channels, height, width)
        x = x[:, 1:3, :, :]  # Select the mask and ground truth channels
        features = self.feature_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        x = x.float()
        x = x.permute(0, 3, 1, 2)  # Permute the dimensions to (batch_size, channels, height, width)
        x = x[:, 0:2, :, :]  # Select the image and mask channels
        x[:, 0, :, :] = x[:, 0, :, :] / 255.0  # normalize the first channel (image) by / 255.0
        features = self.feature_extractor(x)

        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # Reshape the action tensor to match the expected shape
        action = action.view(x.size(0), self.envs.single_action_space.shape[0], self.envs.single_action_space.shape[1], self.envs.single_action_space.shape[2])

        return action, probs.log_prob(action.view(-1, np.prod(self.envs.single_action_space.shape))).sum(
            1), probs.entropy().sum(1), self.critic(features)


if __name__ == "__main__":
    start_time = time.time()
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs_ppo/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if os.name == 'posix':  # macOS and Linux both return 'posix'
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print(x)
        else:
            print("MPS device not found.")
    elif os.name == 'nt':  # Windows returns 'nt'
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        print("Unknown operating system.")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.data_path, args.num_control_points, args.max_iter, args.iou_threshold, args.interval_action_space, args.iou_truncate) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    count_improvement_rwd = 0
    count_rwd = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print(f'Number of iterations: {args.num_iterations}\n')

    for iteration in range(1, args.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            if step % 1000 == 0:
                # show the progress in rounded % of the total steps
                print(f'{round(step / args.num_steps * 100)}% of the total steps completed for iteration {iteration}')

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            for r in reward:
                count_rwd += 1
                treshold = 0.5
                if r > treshold:
                    print(f"reward={r}")
                    count_improvement_rwd += 1
                    for env in envs.envs:
                        if env.rwd() > treshold:
                            img = env.render(mode='rgb_array')
                            img = img.transpose(2, 0, 1)  # Reshape the image to (C, H, W) format
                            writer.add_image(f"charts/obs_eval_{iter}_rwd={env.rwd()}", img, global_step)

            next_done = np.logical_or(terminations, truncations)
            if os.name == 'posix':  # macOS and Linux both return 'posix'
                rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            else:
                rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "iteration" in info and "iou" in info:
                        # print(f"global_step={global_step}, iteration={info['iteration']}, iou={info['iou']}")
                        writer.add_scalar("charts/iteration", info["iteration"], global_step)
                        writer.add_scalar("charts/iou", info["iou"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # show the progress in rounded % of the total iterations
        print(f'--------------------------------{round(iteration / args.num_iterations * 100)}% of the total iterations completed--------------------------------\n')

        # display the time needed for an iteration
        end_time = time.time()
        execution_time = end_time - start_time  # Time in seconds
        hours = execution_time // 3600
        minutes = (execution_time % 3600) // 60
        seconds = (execution_time % 3600) % 60
        print(f"Execution time for iteration {iteration}: {int(hours)}:{int(minutes)}:{int(seconds)}\n")

    print(f'Number of improvements: {count_improvement_rwd} out of {count_rwd} rewards\n')

    if args.save_model:
        model_path = f"runs_ppo/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            eval_episodes=100,
            Model=Agent,
            device=device,
            args=args,
            render=False,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        print(f"average episodic return is {np.mean(episodic_returns)}")

    envs.close()
    writer.close()

    end_time = time.time()
    execution_time = end_time - start_time  # Time in seconds
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = (execution_time % 3600) % 60
    print(f"Execution time: {int(hours)}:{int(minutes)}:{int(seconds)}")

