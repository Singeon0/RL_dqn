import copy

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

from custom_env import process_mask

# Load the data
data_path = "../utah_test_set.h5"
data = h5py.File(data_path, "r")

images = np.array(data['image'])

# Extract labels for all patients
labels = np.array(data['prediction'])

# for i in range(len(images)):
#     plt.imshow(labels[i], cmap='gray')
#     plt.title('mask')
#     plt.show()

idx = 15

# Access and print the label matrix of the 56th patient
# Assuming the first dimension indexes patients
image = images[idx]
label = labels[idx]

plt.imshow(image, cmap='gray')
plt.title('data')
# plt.show()

mask = label

plt.imshow(mask, cmap='gray')
plt.title('mask')
# plt.show()

def apply_mask_to_image(image, mask, intensity):
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

img = apply_mask_to_image(image, mask*255, 0.5)

plt.imshow(img, cmap='gray')
plt.title('Combined Image')
# plt.show()

plt.imshow(mask, cmap='gray')
plt.title('mask')
plt.show()

from scipy.ndimage import zoom


def resize_grayscale_image(image_array):
    """
    Resizes a grayscale image represented as a NumPy array with shape values as 1 and background as 0.

    :param image_array: NumPy array of the grayscale image.
    :return: Resized NumPy array with doubled height and width.
    """
    # Apply the zoom and double the dimensions
    resized_image = zoom(image_array, zoom=2, order=0)  # order=0 means nearest-neighbor interpolation

    # Clip values to ensure the image remains binary (0s and 1s)
    resized_image = np.clip(resized_image, 0, 1)

    return resized_image


mask_2 = resize_grayscale_image(mask)

plt.imshow(mask_2, cmap='gray')
plt.title('mask_super_res')
plt.show()

mask_position = np.where(mask == 1)
mask_position_2 = np.where(mask_2 == 1)

from pygem import FFD

ffd = FFD([2, 2, 1])
ffd.array_mu_x[1, 1] = 1
ffd.array_mu_y[1, 1] = 1
ffd.array_mu_x[0, 0] = -1
ffd.array_mu_y[0, 0] = -1
ffd.array_mu_x[0, 1] = -1
ffd.array_mu_y[0, 1] = 1
ffd.array_mu_x[1, 0] = 1
ffd.array_mu_y[1, 0] = -1
print(ffd)

contour_positions = copy.deepcopy(mask_position_2)
background = copy.deepcopy(img)

# Plot the original background with transparency
plt.imshow(background, cmap='viridis', alpha=0.5)

contour = contour_positions

# contour = get_random_points(contour, n=32)  # retain n points from the contour
# contour = get_evenly_spaced_points_by_distance(contour, n=3)  # retain n points from the contour
# contour = get_evenly_spaced_points(contour, n=32)  # retain n points from the contour
print(f'len contour = {len(contour)}')

# Extract x and y coordinates from the contour positions
x = contour[0]
y = contour[1]

coordinate = np.transpose(np.array([x, y, np.zeros(x.shape[0])]))
coordinate[:, 2] = 1e-6
print(f'shape of coordinate = {coordinate.shape}')
new_contour = np.transpose(ffd(coordinate/mask_2.shape[0]))
new_contour *= mask.shape[0]
print(f'shape of new_contour = {new_contour.shape}')

# check if np.transpose(coordinate) and new_contour are the same
print(np.sum(np.transpose(coordinate) - new_contour))
print(f'diff x axis = {np.sum(np.transpose(coordinate)[0] - new_contour[0])}')
print(f'diff y axis = {np.sum(np.transpose(coordinate)[1] - new_contour[1])}')
print(f'diff z axis = {np.sum(np.transpose(coordinate)[2] - new_contour[2])}')

x_fine, y_fine = new_contour[0], new_contour[1]

shape = process_mask(new_contour, (640, 640))
# print(f'shape[0] = {shape[0]}')
# print(f'shape[1] = {shape[1]}')

# # Plot the parametric spline on top of the background with transparency
plt.scatter(*ffd.control_points()[:, [0, 1]].T * 640)
#plt.scatter(y_fine, x_fine, linewidth=2, alpha=0.8)
plt.imshow(shape, cmap='viridis', alpha=0.5)
plt.show()


