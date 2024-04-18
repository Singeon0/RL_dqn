import copy

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the data
data_path = "utah_test_set.h5"
data = h5py.File(data_path, "r")

images = np.array(data['image'])

# Extract labels for all patients
labels = np.array(data['prediction'])

# for i in range(len(images)):
#     plt.imshow(labels[i], cmap='gray')
#     plt.title('mask')
#     plt.show()

idx = 0

# Access and print the label matrix of the 56th patient
# Assuming the first dimension indexes patients
image = images[idx]
label = labels[idx]

plt.imshow(image, cmap='gray')
plt.title('data')
plt.show()

mask = np.where(label >= 225, float(1), float(0)).astype(image.dtype)

plt.imshow(mask, cmap='gray')
plt.title('mask')
plt.show()

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

plt.imshow(apply_mask_to_image(image, mask*255, 0.2), cmap='gray')
plt.title('Combined Image')
plt.show()


def explore_shape(mask, start_pix):
    """
    Explores the shape made up by 1s in binary image (mask) starting from the provided pixel location (start_pix).
    Uses Breadth-first search (BFS) to explore the shape.

    Args:
        mask (np.array): Binary image where the shape made up by 1s needs to be explored.
        start_pix (tuple): The starting pixel coordinates (row, col)

    Returns:
        list: List of all pixel coordinates that make up the shape
    """
    # Define the relative positions of the 8 neighbors

    temp = mask.copy()

    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    queue = [start_pix]
    shape_pixels = []

    while queue:
        current_pix = queue.pop(0)
        shape_pixels.append(current_pix)
        temp[current_pix[0]][current_pix[1]] = 0  # Mark as visited by setting it to zero
        for dx, dy in neighbors:
            new_x, new_y = current_pix[0] + dx, current_pix[1] + dy
            if (0 <= new_x < temp.shape[0]) and (0 <= new_y < temp.shape[1]) and temp[new_x][new_y] == 1:
                queue.append((new_x, new_y))
                temp[new_x][new_y] = 0  # Mark this pixel as visited

    return shape_pixels, temp


def extract_shape_positions(shape):
    """
    Extracts positions of all shapes included in the mask.

    Args:
        shape (np.ndarray): Binary image where all shapes are made up by 1s.

    Returns:
        list: List of all shapes, where each shape is a list of its pixel coordinates.

    Raises:
        ValueError: If shape is not a numpy ndarray, not of dtype int64, or contains values other than 0 and 1.
    """
    if not isinstance(shape, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if shape.dtype != np.int64:
        raise ValueError("Array dtype must be int64.")
    if not np.all(np.isin(shape, [0, 1])):
        raise ValueError("Array must only contain 0 and 1 values.")

    shape_positions = []
    # Iterate through the mask to find all shapes
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            # If we find an unvisited pixel part of a shape
            if shape[i][j] == 1:
                shape_pixels, shape = explore_shape(shape, (i, j))
                shape_positions.append(shape_pixels)

    return shape_positions

# Apply this function to the test mask to extract the contour positions
mask_position = extract_shape_positions(shape=mask)

from pygem import FFD

ffd = FFD([3, 3, 1])
ffd.array_mu_x[0, 0] = 0
ffd.array_mu_y[0, 0] = 0

def position_to_shape(positions, size):
    shape = np.zeros((size, size))
    for i in range(len(positions[0])):
        x = positions[0][i]
        y = positions[1][i]
        shape[int(x)][int(y)] = 1
    return shape

def plot_parametric_splines(contour_positions, background):
    # Plot the original background with transparency
    plt.imshow(background, cmap='viridis', alpha=0.5)

    for contour in contour_positions:
        # contour = get_random_points(contour, n=32)  # retain n points from the contour
        # contour = get_evenly_spaced_points_by_distance(contour, n=3)  # retain n points from the contour
        # contour = get_evenly_spaced_points(contour, n=32)  # retain n points from the contour
        print(f'len contour = {len(contour)}')

        # Extract x and y coordinates from the contour positions
        x = [pos[1] for pos in contour]
        y = [pos[0] for pos in contour]

        coordinate = np.transpose(np.array([x, y, np.zeros(len(x))]))

        new_contour = np.transpose(ffd(coordinate/640))

        new_contour *= 640

        x_fine, y_fine = new_contour[0], new_contour[1]

        shape = position_to_shape(new_contour, 640)


        # # Plot the parametric spline on top of the background with transparency
        # plt.scatter(*ffd.control_points()[:, [0, 1]].T * 640)
        # plt.scatter(x_fine, y_fine, linewidth=2, alpha=0.8)

        plt.imshow(shape, cmap='viridis', alpha=0.5)




    print(ffd)
    plt.title('FFD')
    plt.axis('off')
    plt.show()

# plot_parametric_splines(contour_positions, contour2)
plot_parametric_splines(mask_position, image)

