# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import copy
import os
import random
import time
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import h5py
import gymnasium as gym
from gym import spaces
from pygem import FFD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def transform_array(input_array):
    """
    Transforms a 1D input array into two 3D arrays.

    This function is used to transform a 1D array of control points into two 3D arrays
    representing the x and y coordinates of the control points in a Free-Form Deformation (FFD) space.
    The z dimension of the output arrays is always 1.

    Parameters:
    input_array (np.array): 1D array of control points. The length of this array should be even.

    Returns:
    tuple: A tuple containing two 3D numpy arrays (array_mu_x, array_mu_y).
           Each array has the shape (dim, dim, 1), where dim is the square root of half the length of the input array.
           array_mu_x contains the x coordinates of the control points.
           array_mu_y contains the y coordinates of the control points.
    """
    # Ensure the length of the input array is appropriate
    if len(input_array) % 2 != 0:
        raise ValueError("Length of input array must be even.")

    num_control_points = len(input_array) // 2
    dim = int(np.sqrt(num_control_points))

    if dim * dim != num_control_points:
        raise ValueError("Number of control points must allow a perfect square dimension.")

    array_mu_x = np.zeros((dim, dim, 1))
    array_mu_y = np.zeros((dim, dim, 1))

    for i in range(dim):
        for j in range(dim):
            idx = 2 * (i * dim + j)
            array_mu_x[i, j, 0] = input_array[idx]
            array_mu_y[i, j, 0] = input_array[idx + 1]

    return array_mu_x, array_mu_y


def apply_mask_to_image(image, mask, intensity=0.85):
    """
    Applies a mask to an image with a given intensity.
    The function combines the image and the mask through a weighted operation, producing a blended output.

    Args:
        image (np.array): The source image, needs to be of type uint8.
        mask (np.array): The mask to overlay. Needs to be of the same type as the image.
            Convert it if necessary, using astype(image.dtype) or astype(np.uint8).
        intensity (float): The intensity or alpha of the mask.

    Returns:
        np.array: The output image, with the mask applied.
    """

    alpha = intensity
    beta = (1.0 - alpha)
    combined = cv2.addWeighted(image, alpha, mask, beta, 0.0)

    return combined


def read_h5_dataset(h5_file_path):
    # Open the .h5 file in read mode
    with h5py.File(h5_file_path, 'r') as h5file:
        # Access the datasets
        images = np.array(h5file['image'])
        predictions = np.array(h5file['prediction'])
        groundtruths = np.array(h5file['groundtruth'])

    return images, predictions, groundtruths


def make_env(env_id, seed, idx, capture_video, run_name, data_path, num_control_points, max_iter, iou_threshold):
    def thunk():
        # Create an instance of the MedicalImageSegmentationEnv
        env = MedicalImageSegmentationEnv(data_path, num_control_points, max_iter, iou_threshold)

        # Set the random seed for the environment
        env.seed(seed)

        # If capture_video is True and this is the first environment (idx is 0), enable video capturing
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # Enable episode statistics recording for the environment
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


class MedicalImageSegmentationEnv(gym.Env):
    def __init__(self, data_path, num_control_points, max_iter, iou_threshold):
        super(MedicalImageSegmentationEnv, self).__init__()

        # Load all images, initial masks, and ground truths
        self.mri_images, self.initial_masks, self.ground_truths = read_h5_dataset(data_path)

        self.num_control_points = num_control_points

        self.num_samples = len(self.mri_images)

        self.current_index = 0

        self.current_mask = copy.deepcopy(self.initial_masks[self.current_index])

        self.iteration = 0
        self.max_iterations = max_iter
        self.iou_threshold = iou_threshold

        # Initialize the FFD object
        self.ffd = FFD([int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)), 1])  # sqrt because i want a square grid

        # Define the action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_control_points, num_control_points, 2),
                                       dtype=np.float32)  # TODO: Space[ActType] warning

        # Define the observation space
        self.observation_space = spaces.Dict({
            'mri_image': spaces.Box(low=0, high=255, shape=self.mri_images[0].shape, dtype=np.int64),
            'current_mask': spaces.Box(low=0, high=1, shape=self.initial_masks[0].shape, dtype=np.int64),
        })

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Select the next set of data
        # In the context of zero-indexed data structures, this operation ensures that self.current_index always lies
        # within the range of valid indices. Once self.current_index reaches the end of the data structure (the index
        # becomes equal to self.num_samples), it wraps around back to the start.
        self.current_index = (self.current_index + 1) % self.num_samples

        # Reset the current mask to the initial mask
        self.current_mask = copy.deepcopy(np.where(self.initial_masks[self.current_index] >= 225, float(1), float(0)).astype('int64'))

        # Reset the FFD object
        self.ffd = FFD([int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)), 1])  # sqrt because i want a square grid

        # Reset the iteration counter
        self.iteration = 0

        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        return {
            'mri_image': self.mri_images[self.current_index],
            'current_mask': self.current_mask,
        }

    def step(self, action):
        # Apply the action to deform the current mask using FFD
        self.current_mask = self._apply_action(action)

        # Compute the reward based on the improvement in IoU
        reward = self._compute_reward()

        # Increment the iteration counter
        self.iteration += 1

        # Check if the episode is done
        done = self._is_done()

        # Create the info dictionary
        info = {
            'iteration': self.iteration,
            'iou': self._compute_iou(self.current_mask, self.ground_truths[self.current_index]),
        }

        # Return the next observation, reward, done flag, and info
        return self._get_observation(), reward, done, info

    def _apply_action(self, action):
        """

        :param action: np.array ; value of new control points provided by the neural network
        :return:
        """
        # Reshape the action to match the expected shape of control points

        self.ffd.array_mu_x, self.ffd.array_mu_y = transform_array(action)

        # Apply the FFD transformation to the current mask
        mask_coordinates = np.where(self.current_mask == 1)  # only keep position of the biggest shape

        # test = np.zeros(self.current_mask.shape, dtype=self.current_mask.dtype)
        # test[mask_coordinates[0].astype(int), mask_coordinates[1].astype(int)] = 1
        # plt.imshow(test, cmap='viridis')
        # plt.title('test')
        # plt.show()

        coordinate = np.transpose(np.array([mask_coordinates[0], mask_coordinates[1], np.zeros(len(mask_coordinates[0]))]))

        new_shape = np.transpose(self.ffd(coordinate / np.shape(self.current_mask)[0]))

        new_shape *= np.shape(self.current_mask)[0]  # TODO if the mask get bigger I will have hole in it

        shape_img = np.zeros(self.current_mask.shape, dtype=self.current_mask.dtype)

        shape_img[new_shape[0].astype(int), new_shape[1].astype(int)] = 1

        # plt.imshow(shape_img, cmap='viridis')
        # plt.title('new shape')
        # plt.show()

        return shape_img

    def _compute_reward(self):
        # Compute the IoU between the deformed mask and the ground truth mask
        iou_deformed = self._compute_iou(self.current_mask, self.ground_truths[self.current_index])
        iou_initial = self._compute_iou(self.initial_masks[self.current_index], self.ground_truths[self.current_index])

        # Compute the reward as the improvement in IoU
        reward = iou_deformed - iou_initial

        return reward

    def _is_done(self):
        # Check if the maximum number of iterations is reached
        if self.iteration >= self.max_iterations:
            return True

        # Check if the desired IoU threshold is achieved
        iou = self._compute_iou(self.current_mask, self.ground_truths[self.current_index])
        if iou >= self.iou_threshold:
            return True

        return False

    def _compute_iou(self, mask1, mask2):
        # Flatten the masks to 1D arrays
        mask1 = mask1.flatten()
        mask2 = mask2.flatten()

        # Compute the Jaccard score (equivalent to IoU) between the two masks
        iou = jaccard_score(mask1, mask2, average='binary')

        return iou

    def render(self, mode='human'):
        # Placeholder for the render method
        pass


class QNetwork(nn.Module):
    """
        A class used to represent the Q-Network for a reinforcement learning task.

        This class is a subclass of PyTorch's nn.Module and is used to define the neural network architecture for a Q-learning algorithm.
        The network takes an observation from the environment as input and outputs a vector of Q-values, one for each possible action.

        Attributes
        ----------
        mri_image_conv : nn.Sequential
            The convolutional layers for processing the MRI image from the observation.
        current_mask_conv : nn.Sequential
            The convolutional layers for processing the current mask from the observation.
        fc : nn.Sequential
            The fully connected layers for processing the concatenated features from the convolutional layers.

        Methods
        -------
        _calculate_conv_output_size(input_shape)
            Calculates the output size of the convolutional layers.
        forward(observations)
            Defines the forward pass of the network.
    """

    def __init__(self, env):
        """
            Constructs all the necessary attributes for the QNetwork object.

            Parameters
            ----------
                env : gym.Env
                    The environment from which the network will receive observations and in which the network's outputs will be used as actions.
        """
        super().__init__()

        # Get the shapes of the MRI image and current mask from the observation space
        mri_image_shape = env.observation_space['mri_image'].shape
        current_mask_shape = env.observation_space['current_mask'].shape

        # Define the convolutional layers for the MRI image
        self.mri_image_conv = nn.Sequential(
            nn.Conv2d(mri_image_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Define the convolutional layers for the current mask
        self.current_mask_conv = nn.Sequential(
            nn.Conv2d(current_mask_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate the flattened size of the feature maps
        mri_image_features = self._calculate_conv_output_size(mri_image_shape)
        current_mask_features = self._calculate_conv_output_size(current_mask_shape)

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(mri_image_features + current_mask_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.shape[0]),
        )

    def _calculate_conv_output_size(self, input_shape):
        # Calculate the output size of the convolutional layers
        conv_output = self.mri_image_conv(torch.zeros(1, *input_shape))
        return int(np.prod(conv_output.size()))

    def forward(self, observations):
        # Extract the MRI image and current mask from the observations
        mri_image = observations['mri_image']
        current_mask = observations['current_mask']

        # Process the MRI image through the convolutional layers
        mri_image_features = self.mri_image_conv(mri_image)
        mri_image_features = mri_image_features.view(mri_image_features.size(0), -1)

        # Process the current mask through the convolutional layers
        current_mask_features = self.current_mask_conv(current_mask)
        current_mask_features = current_mask_features.view(current_mask_features.size(0), -1)

        # Concatenate the features from the MRI image and current mask
        combined_features = torch.cat((mri_image_features, current_mask_features), dim=1)

        # Pass the combined features through the fully connected layers
        q_values = self.fc(combined_features)

        return q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def batch_dict_space(space, n):
    return gym.spaces.Dict({k: gym.spaces.Box(low=np.tile(v.low, (n, 1)),
                                              high=np.tile(v.high, (n, 1)))
                            for k, v in space.spaces.items()})

def batch_box_space(space, n):
    return gym.spaces.Box(low=np.tile(space.low, (n, 1)),
                          high=np.tile(space.high, (n, 1)))


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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

    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name,
               data_path="utah_test_set.h5",
               num_control_points=16,
               max_iter=100,
               iou_threshold=0.9)()

    observation_space = batch_dict_space(env.observation_space, args.num_envs)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,
                  data_path="utah_test_set.h5",
                  num_control_points=16,
                  max_iter=100,
                  iou_threshold=0.9) for i in range(args.num_envs)],
        observation_space=batch_dict_space(env.observation_space, args.num_envs),
        action_space=batch_box_space(env.action_space, args.num_envs)
    )

    # Create an instance of the QNetwork with the custom environment
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(obs)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            data_path="utah_test_set.h5",
            num_control_points=16,
            max_iter=100,
            iou_threshold=0.9,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
