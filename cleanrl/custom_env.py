import copy
import warnings

import cv2
import gymnasium as gym
from collections import Counter
from scipy.ndimage import binary_closing, zoom
import h5py
import matplotlib.pyplot as plt
import numpy as np
from gymnasium  import spaces
from pygem import FFD
from sklearn.metrics import jaccard_score

warnings.filterwarnings("ignore", category=DeprecationWarning)


def transform_array(input_array):  # TODO: check if this is the right way to do it because action came from  Box(low=-1, high=1, shape=(int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)), 2)

    array_mu_x = input_array[:, :, 0:1]
    array_mu_y = input_array[:, :, 1:2]

    return array_mu_x, array_mu_y


def apply_mask_to_image(image, mask, intensity=0.5):
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

    # Ensure the image and mask are of type float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    # plt.imshow(image, cmap='gray')
    # plt.title('image')
    # plt.show()
    #

    # plt.imshow(mask, cmap='gray')
    # plt.title('mask')
    # plt.show()

    alpha = intensity
    beta = (1.0 - alpha)
    combined = cv2.addWeighted(image, alpha, mask, beta, 0.0)

    # plt.imshow(combined, cmap='gray')
    # plt.title('combined')
    # plt.show()

    return combined


def process_mask(data, img_size):
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
    print(f'new_shape_int: {new_shape_int.shape}')

    if new_shape_int.shape[1] > 1:

        # Transpose the array and count duplicates
        arr = np.transpose(new_shape_int)
        tuples = [tuple(row) for row in arr]
        counts = Counter(tuples)
        duplicates = {column: count for column, count in counts.items() if count > 1}

        # Increase the size of the mask until there are no duplicates
        i = 1
        while len(duplicates) > 0 and i <= 10:
            new_shape_int = (new_shape*i).astype(int)
            arr = np.transpose(new_shape_int)
            tuples = [tuple(row) for row in arr]
            counts = Counter(tuples)
            duplicates = {column: count for column, count in counts.items() if count > 1}
            i += 1

        # Create a new image of the appropriate size and apply the mask
        new_size = img_size[0] * i, img_size[1] * i
        shape_img = np.zeros(new_size, dtype=np.float32)

        # Ensure the indices are within the bounds of the array
        indices_0 = new_shape_int[0].astype(int)
        indices_1 = new_shape_int[1].astype(int)

        # Check if indices are within the bounds
        if not np.max(indices_0) < shape_img.shape[0] and np.max(indices_1) < shape_img.shape[1]:
            # print("Indices are out of bounds of the array shape_img.")
            temp = []
            for j in range(len(indices_0)):
                if 0 <= new_shape_int[0][j] < (shape_img.shape[0] - 1) and 0 <= new_shape_int[1][j] < (
                        shape_img.shape[1] - 1):
                    # adding the element corresponding as the index as a tuple in temp
                    temp.append((indices_0[j], indices_1[j]))
            # converting the list of tuples to a numpy array of shape (2, len(temp))
            new_shape_int = np.array(temp).T

        # handle IndexError
        try:
            if new_shape_int.size > 0:  # Check if new_shape_int is not empty
                shape_img[new_shape_int[0].astype(int), new_shape_int[1].astype(int)] = 1
            else:
                return np.zeros(img_size, dtype=np.float32)
        except IndexError:
            # print('PROBLEM')
            temp = []
            # print(f'SHAPE new_shape_int: {new_shape_int.shape}\nnew_shape_int: {new_shape_int}')
            for j in range(len(new_shape_int[0])):
                if 0 <= new_shape_int[0][j] < (shape_img.shape[0] - 1) and 0 <= new_shape_int[1][j] < (
                        shape_img.shape[1] - 1):
                    # adding the element corresponding as the index as a tuple in temp
                    temp.append((indices_0[j], indices_1[j]))
            # converting the list of tuples to a numpy array of shape (2, len(temp))
            new_shape_int = np.array(temp).T
            if len(temp) > 0:
                shape_img[new_shape_int[0].astype(int), new_shape_int[1].astype(int)] = 1

        # Apply morphological closing to close small gaps in the contours
        closed_image = binary_closing(shape_img, structure=np.ones((3, 3)))

        # Resize the closed image to the original img_size
        resized_image = cv2.resize(closed_image.astype(float), (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)

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
    def __init__(self, data_path, num_control_points, max_iter, iou_threshold):
        super(MedicalImageSegmentationEnv, self).__init__()

        # Load all images, initial masks, and ground truths
        self.mri_images, self.initial_masks, self.ground_truths = read_h5_dataset(data_path)

        self.np_random = None

        self.num_control_points = num_control_points

        self.num_samples = len(self.mri_images)

        self.current_index = 0

        self.current_mask = copy.deepcopy(self.initial_masks[self.current_index])

        plt.imshow(self.current_mask, cmap='gray')
        plt.title('initial mask')
        # plt.show()

        plt.imshow(self.ground_truths[self.current_index], cmap='gray')
        plt.title('ground truth')
        # plt.show()

        self.iteration = 0
        self.max_iterations = max_iter
        self.iou_threshold = iou_threshold

        # Initialize the FFD object
        self.ffd = FFD([int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)),
                        1])  # sqrt because i want a square grid

        # Define the action space
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(int(np.sqrt(self.num_control_points)), int(np.sqrt(self.num_control_points)), 2),
                                       dtype=np.float32)
        """
        # Example of how to access value of control points from the action array
        array_mu_x = self.action_space[:,:,0:1]
        array_mu_y = self.action_space[:,:,1:2]
        """

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.mri_images[0].shape[0], self.mri_images[0].shape[0], 1),
                                            dtype=np.float32)

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

    def _get_observation(self):
        # Apply the current mask to the MRI image
        observation = apply_mask_to_image(self.mri_images[self.current_index], self.current_mask)
        return observation

    def step(self, action):
        # Apply the action to deform the current mask using FFD
        self.current_mask = self._apply_action(action)

        # Compute the reward based on the improvement in IoU
        reward = self._compute_reward()

        print(f'iteration: {self.iteration}, reward: {reward}, iou: {self._compute_iou(self.current_mask, self.ground_truths[self.current_index])}')

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

        # upscale the mask
        mask = resize_grayscale_image(self.current_mask, upscale_factor=2)

        # Apply the FFD transformation to the current mask
        mask_coordinates = np.where(mask == 1)  # only keep position of the biggest shape

        # test = np.zeros(self.current_mask.shape, dtype=self.current_mask.dtype)
        # test[mask_coordinates[0].astype(int), mask_coordinates[1].astype(int)] = 1
        # plt.imshow(test, cmap='viridis')
        # plt.title('test')
        # plt.show()

        coordinate = np.transpose(np.array([mask_coordinates[0], mask_coordinates[1], np.zeros(len(mask_coordinates[0]))]))

        # TO SOLVE THE STUPID FFD BUG WHERE IF THE Z AXIS IS 0 IT DOESN'T WORK
        coordinate[:, 2] = 1e-3

        new_shape = np.transpose(self.ffd(coordinate / np.shape(mask)[0]))  # divide by the size of the mask upscaled to have pixels position between 0 and 1, its like putting pixel in an intermediate space

        new_shape *= np.shape(self.current_mask)[0]  # multiply by the size of the mask to have the pixel position in the image space

        shape_img = process_mask(new_shape, self.current_mask.shape)

        plt.imshow(shape_img, cmap='gray')
        plt.title('new shape')
        # plt.show()

        return shape_img

    def _compute_reward(self):
        # Compute the IoU between the deformed mask and the ground truth mask
        iou_deformed = self._compute_iou(self.current_mask, self.ground_truths[self.current_index])
        iou_initial = self._compute_iou(self.initial_masks[self.current_index], self.ground_truths[self.current_index])

        # Compute the reward as the improvement in IoU
        reward = iou_deformed - iou_initial

        return reward

    def _is_terminated(self):
        # Calculate the total number of pixels in the current mask and the ground truth
        total_pixels_mask = np.sum(self.current_mask)
        total_pixels_ground_truth = np.sum(self.ground_truths[self.current_index])

        # Stop if the number of pixels in the mask is less than x% of the number of pixels in the ground truth
        if total_pixels_mask < 0.15 * total_pixels_ground_truth:
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
        # Placeholder for the render method
        pass
