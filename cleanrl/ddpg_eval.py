import copy
from typing import Callable

import torch
from pathlib import Path


def evaluate(
    model_path: str,
    make_env: Callable,
    eval_episodes: int,
    run_name: str,
    Model: tuple,
    device: torch.device = torch.device("cpu"),
    exploration_noise: float = 0.1,
):
    env = make_env()
    actor = Model[0](env).to(device)
    qf = Model[1](env).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()
    # note: qf is not used in this script

    episodic_returns = []
    obs_env = []
    while len(episodic_returns) < eval_episodes:
        obs, _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            with torch.no_grad():
                temp = copy.deepcopy(obs)
                temp = torch.from_numpy(temp[:, :, 0:2]).permute(2, 0, 1).unsqueeze(0)
                action = actor(torch.Tensor(temp).to(device))
                action += torch.normal(0, actor.action_scale * exploration_noise)
                action = action.cpu().numpy().reshape(env.action_space.shape).clip(env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated or truncated:
                img = env.render(mode='rgb_array')
                obs_env.append(img)
                episode_return += reward
            obs = next_obs

        print(f"eval_episode={len(episodic_returns)}, episodic_return={episode_return}")
        episodic_returns.append(episode_return)

    return episodic_returns, obs_env


if __name__ == "__main__":
    from cleanrl.ddpg_eval import evaluate
    from cleanrl.ddpg_continuous_action_images_segmentation import Actor, QNetwork, make_env

    data_path = Path('../cleanrl_utils') / 'synthetic_ds' / 'synthetic_dataset.h5'
    num_control_points = 16
    max_iter = 100
    iou_threshold = 0.85
    interval_action_space = 0.25

    model_path = Path(
        "runs/ddpg_continuous_action_images_segmentation__1__1714473083/ddpg_continuous_action_images_segmentation"
        ".cleanrl_model")

    evaluate(
        model_path,
        make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space),
        eval_episodes=25,
        run_name="eval",
        Model=(Actor, QNetwork),
        device="cpu",
    )