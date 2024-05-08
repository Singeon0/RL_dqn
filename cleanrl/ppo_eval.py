import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")



def evaluate(
    model_path: str,
    make_env: Callable,
    eval_episodes: int,
    Model: torch.nn.Module,
    data_path, num_control_points, max_iter, iou_threshold, interval_action_space, iou_truncate,
    device: torch.device = torch.device("cpu"),
    render: bool = False,
):
    if os.name == 'posix':  # macOS and Linux both return 'posix'
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print(x)
        else:
            print("MPS device not found.")
    elif os.name == 'nt':  # Windows returns 'nt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("Unknown operating system.")

    envs = gym.vector.SyncVectorEnv([make_env(data_path, num_control_points, max_iter, iou_threshold, interval_action_space, iou_truncate)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    total_episodes = eval_episodes
    while len(episodic_returns) < total_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if render:
            for env in envs.envs:
                env.render()
        if "final_info" in infos:
                for info in infos["final_info"]:
                    if "iteration" not in info or "iou" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, iteration={info['iteration']}, iou={info['iou']}")
                    episodic_returns += [info["iou"]]
        obs = next_obs
    return episodic_returns

if __name__ == "__main__":
    from ppo import Agent, make_env
    from pathlib import Path
    from ppo import Args
    import tyro

    args = tyro.cli(Args)

    cu = True

    model_path = Path(
        "runs_ppo/Seg_PPO-v0__ppo__1__1715081971/ppo.cleanrl_model")

    episodic_returns = evaluate(
        model_path,
        make_env,
        eval_episodes=10,
        Model=Agent,
        device="cuda" if torch.cuda.is_available() and cu else "cpu",
        args=args,
        render=True
    )

    for idx, episodic_return in enumerate(episodic_returns):
        print(f"eval_episode={idx}, episodic_return={episodic_return}")
    print(f"average episodic return is {np.mean(episodic_returns)}")
