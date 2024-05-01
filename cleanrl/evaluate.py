import torch
from ddpg_continuous_action_images_segmentation import Actor, QNetwork, make_env
from cleanrl_utils.evals.ddpg_eval import evaluate
from pathlib import Path

def evaluate_model(model_path, data_path, num_control_points, max_iter, iou_threshold, eval_episodes=10, device="cpu"):
    episodic_returns = evaluate(
        model_path,
        make_env(data_path, num_control_points, max_iter, iou_threshold),
        eval_episodes=eval_episodes,
        run_name="eval",
        Model=(Actor, QNetwork),
        device=device,
    )
    for idx, episodic_return in enumerate(episodic_returns):
        print(f"eval_episode={idx}, episodic_return={episodic_return}")
    return episodic_returns


if __name__ == "__main__":
    data_path = Path('..') / 'synthetic_ds' / 'synthetic_dataset.h5'
    num_control_points = 16
    max_iter = 100
    iou_threshold = 0.85
    from pathlib import Path

    model_path = Path(
        "runs/ddpg_continuous_action_images_segmentation__1__1714473083/ddpg_continuous_action_images_segmentation.cleanrl_model")
    episodic_returns = evaluate_model(model_path, data_path, num_control_points, max_iter, iou_threshold)