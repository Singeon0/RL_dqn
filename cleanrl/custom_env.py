import copy
import random
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from pygem import FFD
from sklearn.metrics import jaccard_score

from utils import create_action_matrix, transform_array, process_mask, read_h5_dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MedicalImageSegmentationEnv(gym.Env):
    def __init__(self, data_path, num_control_points, max_iter, iou_threshold, interval_action_space=0.125):
        super(MedicalImageSegmentationEnv, self).__init__()

        # Load all images, initial masks, and ground truths
        self.mri_images, self.initial_masks, self.ground_truths = read_h5_dataset(data_path)

        self.np_random = None

        self.num_control_points = num_control_points

        self.num_samples = len(self.mri_images)

        self.current_index = 0

        self.current_mask = copy.deepcopy(self.initial_masks[self.current_index])

        self.iteration = 0
        self.max_iterations = max_iter
        self.iou_threshold = iou_threshold

        # Initialize the FFD object
        self.ffd = FFD([int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)),
                        1])  # sqrt because i want a square grid

        self.interval_action_space = interval_action_space

        # Define the action space
        self.action_space = spaces.Box(low=-interval_action_space, high=interval_action_space, shape=(int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)), 2),
                                       dtype=np.float16)
        """
        # Example of how to access value of control points from the action array
        array_mu_x = self.action_space[:,:,0:1]
        array_mu_y = self.action_space[:,:,1:2]
        """

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.mri_images[0].shape[0], self.mri_images[0].shape[0], 3),  # Change the 1 to 3 bc the observation is composed of image, mask and ground truth
                                            dtype=np.float32)

        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Action space shape: {self.action_space.shape}")


    def action_sample(self, percentage=0.05, interval_action_space=0.125):
        """
        Randomly sample an action from the action space, with a small chance of expanding the mask.
        Args:
            percentage: The probability of expanding the mask.
            interval_action_space: The interval of the action space.

        Returns: action parameters

        """
        if random.random() <= percentage:  # 5% chance of expansion
            return create_action_matrix(self.num_control_points, interval_action_space)
        else:
            return self.action_space.sample()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        # Select the next set of data
        # In the context of zero-indexed data structures, this operation ensures that self.current_index always lies
        # within the range of valid indices. Once self.current_index reaches the end of the data structure (the index
        # becomes equal to self.num_samples), it wraps around back to the start.
        self.current_index = (self.current_index + 1) % self.num_samples

        # Reset the current mask to the initial mask
        self.current_mask = self.initial_masks[self.current_index].copy()

        # Reset the FFD object
        self.ffd = FFD([int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)),
                        1])  # sqrt because i want a square grid

        # Reset the iteration counter
        self.iteration = 0

        # Seed the environment if a seed is provided
        if seed is not None:
            self.seed(seed)

        # Return the initial observation
        return self._get_observation(), {}

    def _get_observation(self):  # TODO : maybe i should had the information if the mask get bigger or smaller ?
        # Apply the current mask to the MRI image
        observation = np.dstack((self.mri_images[self.current_index], self.current_mask, self.ground_truths[self.current_index]))
        """
        # Assuming 'observation' is your stacked numpy array
        mri_image = observation[:, :, 0]
        current_mask = observation[:, :, 1]
        ground_truth = observation[:, :, 2]
        """
        return observation


    def step(self, action):
        # Apply the action to deform the current mask using FFD
        self.current_mask = self._apply_action(action)  # TODO what append is the number of control points increase ? More the control points less the shape reduce in size ?

        # Compute the reward based on the improvement in IoU
        reward = self._compute_reward()  # TODO change the way of calculate the reward because the reward it is always negative

        # print(f'iteration: {self.iteration}, reward: {reward}, iou: {self._compute_iou(self.current_mask, self.ground_truths[self.current_index])}')

        # Increment the iteration counter
        self.iteration += 1

        # Check if the episode is terminated or truncated
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Get the current observation
        observation = self._get_observation()

        # Create the info dictionary
        info = {'iteration': self.iteration,
            'iou': self._compute_iou(self.current_mask, self.ground_truths[self.current_index]), }

        # Store the final observation in the info dictionary if the episode is terminated or truncated
        if terminated or truncated:
            info['final_observation'] = observation

        # Return the next observation, reward, done flag, and info
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        """

        :param action: np.array ; value of new control points provided by the neural network
        :return:
        """
        # Reshape the action to match the expected shape of control points
        self.ffd.array_mu_x, self.ffd.array_mu_y = transform_array(action)

        #  OVERIDE FFD PARAM
        if False:
            self.ffd.array_mu_x[1, 1] = 0.09
            self.ffd.array_mu_y[1, 1] = 0.09
            self.ffd.array_mu_x[0, 0] = -0.09
            self.ffd.array_mu_y[0, 0] = -0.09
            self.ffd.array_mu_x[0, 1] = -0.09
            self.ffd.array_mu_y[0, 1] = 0.09
            self.ffd.array_mu_x[1, 0] = 0.09
            self.ffd.array_mu_y[1, 0] = -0.09

        # upscale the mask
        # mask = resize_grayscale_image(self.current_mask, upscale_factor=2)  # TODO : actually upscale the mask create a problem of position of the mask ; it will be necessary only if the size of the mask increase drastically, so if parameters in action space are big

        mask = copy.deepcopy(self.current_mask)

        # Apply the FFD transformation to the current mask
        mask_coordinates = np.where(mask == 1)  # only keep position of the biggest shape

        coordinate = np.transpose(np.array([mask_coordinates[0], mask_coordinates[1], np.zeros(len(mask_coordinates[0]))]))

        # TO SOLVE THE STUPID FFD BUG WHERE IF THE Z AXIS IS 0 IT DOESN'T WORK
        coordinate[:, 2] = 1e-3

        new_shape = np.transpose(self.ffd(coordinate / np.shape(mask)[0]))  # divide by the size of the mask upscaled to have pixels position between 0 and 1, its like putting pixel in an intermediate space

        new_shape *= np.shape(self.current_mask)[0]  # multiply by the size of the mask to have the pixel position in the image space

        shape_img = process_mask(new_shape, self.current_mask.shape)

        return shape_img

    def _compute_reward(self):
        mask = self.current_mask
        ground_truth = self.ground_truths[self.current_index]

        iou = np.sum(mask & ground_truth) / np.sum(mask | ground_truth)
        excess = np.sum(mask & ~ground_truth) / np.sum(mask)
        reward = iou - excess  # TODO : room for improvement?
        return reward ** 2


    def _is_terminated(self):
        # Calculate the total number of pixels in the current mask and the ground truth
        total_pixels_mask = np.sum(self.current_mask)
        total_pixels_ground_truth = np.sum(self.ground_truths[self.current_index])

        # Stop if the number of pixels in the mask is less than x% of the number of pixels in the ground truth
        if total_pixels_mask < 0.01 * total_pixels_ground_truth:
            return True

        # Check if the desired IoU threshold is achieved
        iou = self._compute_iou(self.current_mask, self.ground_truths[self.current_index])
        return iou >= self.iou_threshold

    def _is_truncated(self):
        # Check if the maximum number of iterations is reached
        return self.iteration >= self.max_iterations

    def _compute_iou(self, mask1, mask2):
        # Flatten the masks to 1D arrays
        mask1 = mask1.flatten()
        mask2 = mask2.flatten()

        # Compute the Jaccard score (equivalent to IoU) between the two masks
        iou = jaccard_score(mask1, mask2, average='binary')

        return iou

    def render(self, mode='human'):
        if mode == 'human':
            self._render_frame()
        elif mode == 'rgb_array':
            return self._render_frame(mode=mode)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render_frame(self, mode='human'):
        # Normalize the images to have values between 0 and 1
        mri_image = self.mri_images[self.current_index].astype(np.float32) / 255.0
        current_mask = self.current_mask.astype(np.float32)
        ground_truth = self.ground_truths[self.current_index].astype(np.float32)

        if mode == 'human':
            # Create a figure
            fig, ax = plt.subplots(figsize=(5, 5))

            # Display the MRI image in grayscale
            ax.imshow(mri_image, cmap='viridis', alpha=0.8)
            title = f'At it {self.iteration}, reward is {np.round(self._compute_reward(), 3)}'
            ax.set_title(title)
            ax.axis('off')

            # Overlay the ground truth mask in red with higher transparency # TODO
            # ax.imshow(np.ma.masked_where(ground_truth == 0, ground_truth), cmap='Reds', alpha=0.6)

            # Overlay the current mask in green with higher transparency
            ax.imshow(np.ma.masked_where(current_mask == 0, current_mask), cmap='viridis', alpha=0.6)

            # Adjust the layout
            plt.tight_layout()

            # Display the plot
            plt.show(block=False)
            plt.pause(1)
            plt.close()


        elif mode == 'rgb_array':
            # Multiply each image by its desired intensity
            mri_image_intensity = mri_image * 100
            ground_truth_intensity = ground_truth * 150
            current_mask_intensity = current_mask * 200

            # Stack the images along the third axis (channel dimension)
            rgb_array = np.dstack((mri_image_intensity, ground_truth_intensity, current_mask_intensity))

            return rgb_array
