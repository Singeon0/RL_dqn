import copy
import warnings
import random

import gymnasium as gym
import cv2
from collections import Counter
from scipy.ndimage import binary_closing, zoom
import numpy as np
import h5py
import matplotlib.pyplot as plt
from gymnasium  import spaces
from pygem import FFD
from sklearn.metrics import jaccard_score

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_action_matrix(n_control_points, corner_value):
    """
    Create a matrix with dimensions based on the square root of n_control_points,
    and set corner values with alternating pattern for mu_x and mu_y.

    Args:
    - n_control_points (int): Total number of control points. Must be a perfect square.
    - corner_value (float): Base value to use for setting corner values.

    Returns:
    - np.array: A numpy array with the specified corner values set.
    """
    # Calculate the dimensions of the matrix
    lp = int(np.sqrt(n_control_points))
    if lp * lp != n_control_points:
        raise ValueError("n_control_points must be a perfect square")

    # Initialize the array
    action = np.zeros((lp, lp, 2))

    # Set the corners to specified values based on your example
    # mu_x assignments
    action[lp - 1, lp - 1, 0] = corner_value  # Bottom-right
    action[0, 0, 0] = -corner_value  # Top-left
    action[0, lp - 1, 0] = -corner_value  # Top-right
    action[lp - 1, 0, 0] = corner_value  # Bottom-left

    # mu_y assignments
    action[lp - 1, lp - 1, 1] = corner_value  # Bottom-right
    action[0, 0, 1] = -corner_value  # Top-left
    action[0, lp - 1, 1] = corner_value  # Top-right
    action[lp - 1, 0, 1] = -corner_value  # Bottom-left

    return action


def transform_array(input_array):
    array_mu_x = input_array[:, :, 0:1]
    array_mu_y = input_array[:, :, 1:2]

    return array_mu_x, array_mu_y


def apply_mask_to_image(image, mask, intensity=0.5):
    """
    Applies a mask to an image with a given intensity.
    The function combines the image and the mask through a weighted operation, producing a blended output.

    Args:
        image (np.array): The source image, needs to be of type float32.
        mask (np.array): The mask to overlay. Needs to be of the same type as the image.
            Convert it if necessary, using astype(image.dtype) or astype(np.float32).
        intensity (float): The intensity or alpha of the mask.

    Returns:
        np.array: The output image, with the mask applied.
    """

    # Ensure the image and mask are of type float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    alpha = intensity
    beta = (1.0 - alpha)
    combined = cv2.addWeighted(image, alpha, mask*255, beta, 0.0)  # Multiply the mask by 255 to have the same scale as the image (mask is composed of 1 and 0)
    # transform the combined image to float32
    combined = combined.astype(np.float32)
    return combined


def process_mask(data, img_size=(640, 640)):
    """
    Process a mask to remove duplicates, close gaps, and resize to the original image size.

    Parameters:
    data (np.array): A 3D numpy array representing the mask to be processed.
    img_size (tuple): A tuple representing the size of the original image.

    Returns:
    np.array: The processed mask, resized to the original image size.
    """
    # Delete data in the z axis
    new_shape = np.delete(data, 2, axis=0)
    new_shape_int = new_shape.astype(int)

    if new_shape_int.shape[1] > 1:

        # Transpose the array and count duplicates
        arr = np.transpose(new_shape_int)
        tuples = [tuple(row) for row in arr]
        counts = Counter(tuples)
        duplicates = {column: count for column, count in counts.items() if count > 1}

        # Increase the size of the mask until there are no duplicates
        i = 1
        while len(duplicates) > 0 and i <= 10:  # Limit the number of iterations to prevent infinite loops and crashes
            new_shape_int = (new_shape*i).astype(int)
            arr = np.transpose(new_shape_int)
            tuples = [tuple(row) for row in arr]
            counts = Counter(tuples)
            duplicates = {column: count for column, count in counts.items() if count > 1}
            if len(duplicates) > 0:
                i += 1


        # Create a new image of the appropriate size and apply the mask
        new_size = img_size[0] * i, img_size[1] * i
        shape_img = np.zeros(new_size, dtype=np.float32)

        # Ensure the indices are within the bounds of the array
        x = new_shape_int[0]  # x coordinates
        y = new_shape_int[1]  # y coordinates

        for j in range(len(x)):
            if 0 <= x[j] < new_size[0] and 0 <= y[j] < new_size[1]:
                shape_img[x[j], y[j]] = 1

        # Apply morphological closing to close small gaps in the contours
        closed_image = binary_closing(shape_img, structure=np.ones((3, 3)))

        # Resize the closed image to the original img_size
        try:
            resized_image = cv2.resize(closed_image.astype(float), (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        except cv2.error:
            print('ERROR')
            print(new_shape_int)
            print(img_size)
            plt.imshow(shape_img, cmap='gray')
            plt.title('Shape Image')
            plt.show()

        resized_image = binary_closing(resized_image, structure=np.ones((3, 3)))

        # Threshold the resized image to convert all positive values to 1 and keep zero as 0
        resized_image[resized_image > 0] = 1

    else:
        shape_img = np.zeros(img_size, dtype=np.float32)
        resized_image = shape_img

    return resized_image


def read_h5_dataset(h5_file_path):
    # Open the .h5 file in read mode
    with h5py.File(h5_file_path, 'r') as h5file:
        # Access the datasets
        images = np.array(h5file['image'])
        predictions = np.array(h5file['prediction'])
        groundtruths = np.array(h5file['groundtruth'])

    return images, predictions, groundtruths


def resize_grayscale_image(image_array, upscale_factor=2):
    """
    Resizes a grayscale image represented as a NumPy array with shape values as 1 and background as 0.

    :param image_array: NumPy array of the grayscale image.
    :param upscale_factor: The factor by which to upscale the image.
    :return: Resized NumPy array with doubled height and width.
    """
    # Apply the zoom and double the dimensions
    resized_image = zoom(image_array, zoom=upscale_factor, order=0)  # order=0 means nearest-neighbor interpolation

    # Clip values to ensure the image remains binary (0s and 1s)
    resized_image = np.clip(resized_image, 0, 1)

    return resized_image


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


    def action_sample(self, percentage=0.05):
        """
        Randomly sample an action from the action space, with a small chance of expanding the mask.
        Args:
            percentage: The probability of expanding the mask.

        Returns: action parameters

        """
        if random.random() <= percentage:  # 5% chance of expansion
            return create_action_matrix(self.num_control_points, self.interval_action_space)
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
        reward = iou ** 2 - excess
        return reward


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
        mri_image = self.mri_images[self.current_index] / 255.0
        current_mask = self.current_mask.astype(np.float32)
        ground_truth = self.ground_truths[self.current_index].astype(np.float32)

        if mode == 'human':
            # Create a figure
            fig, ax = plt.subplots(figsize=(5, 5))

            # Display the MRI image in grayscale
            ax.imshow(mri_image, cmap='gray')
            title = f'At it {self.iteration}, reward is {np.round(self._compute_reward(), 3)}'
            ax.set_title(title)
            ax.axis('off')

            # Overlay the current mask in green
            ax.imshow(np.ma.masked_where(current_mask == 0, current_mask), cmap='Greens', alpha=0.5)

            # Overlay the ground truth mask in red
            ax.imshow(np.ma.masked_where(ground_truth == 0, ground_truth), cmap='Reds', alpha=0.5)

            # Adjust the layout
            plt.tight_layout()

            # Display the plot
            plt.show(block=False)
            plt.pause(0.09)
            plt.close()

        elif mode == 'rgb_array':
            # Multiply each image by its desired intensity
            mri_image_intensity = mri_image * 100
            ground_truth_intensity = ground_truth * 150
            current_mask_intensity = current_mask * 200

            # Stack the images along the third axis (channel dimension)
            rgb_array = np.dstack((mri_image_intensity, ground_truth_intensity, current_mask_intensity))

            return rgb_array
