from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    args=None
):
    print(device)

    envs = gym.vector.SyncVectorEnv([make_env(args.data_path, args.num_control_points, args.max_iter, args.iou_threshold, args.interval_action_space, args.iou_truncate)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    total_episodes = eval_episodes
    while len(episodic_returns) < total_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        for env in envs.envs:
            env.render()
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
        # for idx, env in enumerate(envs.envs):
          #  print(f'Env_{idx} iter = ({env.iteration}//{env.max_iterations})\n')
        # print(f"\n------------------------Progress: {len(episodic_returns)}/{total_episodes} episodes completed------------------------\n")

    return episodic_returns

if __name__ == "__main__":
    from ppo import Agent, make_env
    from pathlib import Path
    from ppo import Args
    import tyro

    args = tyro.cli(Args)

    cu = True

    model_path = Path(
        "runs_ppo/Seg_PPO-v0__ppo__1__1715019751/ppo.cleanrl_model")

    episodic_returns = evaluate(
        model_path,
        make_env,
        eval_episodes=10,
        Model=Agent,
        device="cuda" if torch.cuda.is_available() and cu else "cpu",
        args=args
    )

    for idx, episodic_return in enumerate(episodic_returns):
        print(f"eval_episode={idx}, episodic_return={episodic_return}")
    print(f"average episodic return is {np.mean(episodic_returns)}")
