import cv2
from collections import Counter
from scipy.ndimage import binary_closing, zoom
import numpy as np
import h5py
import matplotlib.pyplot as plt


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