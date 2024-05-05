from dataclasses import dataclass
from pathlib import Path


@dataclass
class Args:
    # Algorithm specific arguments
    data_path: Path = Path('..') / 'synthetic_ds' / 'synthetic_dataset.h5'
    """Path to the synthetic dataset"""
    num_control_points: int = 4
    """Number of control points"""
    max_iter: int = 1
    """Maximum number of iterations"""
    iou_threshold: float = 0.8
    """Intersection over Union (IoU) threshold"""
    interval_action_space: float = 0.15
    """Interval of the action space"""
    env_id: str = "Image_Segmentation-v1"
    """the environment id"""
    total_timesteps: int = int(1e2)
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
    learning_starts: int = int(5e1)
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Common arguments
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