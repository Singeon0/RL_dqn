# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from nn_architectures import Actor, QNetwork
from custom_env import MedicalImageSegmentationEnv
from hyper_param import Args


def make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space):
    def thunk():
        env = MedicalImageSegmentationEnv(data_path, num_control_points, max_iter, iou_threshold, interval_action_space)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__CP{args.num_control_points}__AS{args.interval_action_space}__it{args.max_iter}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
                   name=run_name, monitor_gym=True, save_code=True, )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])), )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Device: {device}")

    # env setup
    env = make_env(args.data_path, args.num_control_points, args.max_iter, args.iou_threshold,
                   args.interval_action_space)()

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device,
                      handle_timeout_termination=False, )
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

        rnd = True  # to track if action came from sample or actor

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = np.round(env.action_sample(percentage=0.05, interval_action_space=0.15), 4)
        else:
            with torch.no_grad():
                temp = copy.deepcopy(obs)
                temp = torch.from_numpy(temp[:, :, 0:2]).permute(2, 0, 1).unsqueeze(0)
                action = actor(torch.Tensor(temp).to(device))
                action += torch.normal(0, actor.action_scale * args.exploration_noise)
                action = action.cpu().numpy().reshape(env.action_space.shape).clip(env.action_space.low,
                                                                                   env.action_space.high)  # I MODIFY THAT
                rnd = False

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
                temp_actor = temp_actor[:, :, :, 0:2].permute(0, 3, 1, 2)  # To have the right shape for torch (batch, channel, height, width)
                next_state_actions = target_actor(temp_actor)
                temp_qf1 = copy.deepcopy(data.next_observations)
                temp_qf1 = temp_qf1[:, :, :, [1, 2]].permute(0, 3, 1, 2)
                qf1_next_target = qf1_target(temp_qf1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

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

        episodic_returns, obs_env = evaluate(model_path,
                                             make_env(args.data_path, args.num_control_points, args.max_iter,
                                                      args.iou_threshold, args.iou_threshold), eval_episodes=1000,
                                             run_name="eval", Model=(Actor, QNetwork),
                                             device="cuda" if torch.cuda.is_available() else "cpu", )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            print(f"eval_episode={idx}, episodic_return={episodic_return}")
            img = obs_env[idx]
            img = img.transpose(2, 0, 1)  # Reshape the image to (C, H, W) format
            if idx % 100 == 0:
                writer.add_image(f"charts/obs_eval{idx}", img, global_step)

    env.close()
    writer.close()

    end_time = time.time()
    execution_time = end_time - start_time  # Time in seconds
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = (execution_time % 3600) % 60

    print(f"Execution time: {int(hours)}:{int(minutes)}:{int(seconds)}")
