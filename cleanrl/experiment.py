import copy

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

from custom_env import process_mask

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

idx = 15

# Access and print the label matrix of the 56th patient
# Assuming the first dimension indexes patients
image = images[idx]
label = labels[idx]

plt.imshow(image, cmap='gray')
plt.title('data')
plt.show()

mask = np.where(label >= 225, float(1), float(0)).astype(image.dtype)

plt.imshow(label, cmap='gray')
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

img = apply_mask_to_image(image, label*255, 0.5)

plt.imshow(img, cmap='gray')
plt.title('Combined Image')
plt.show()



# Apply this function to the test mask to extract the contour positions
mask_position = np.where(label == 1)

from pygem import FFD

ffd = FFD([3, 3, 1])
ffd.array_mu_x[0, 0] = -2
ffd.array_mu_y[0, 0] = 5

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

    contour = contour_positions

    # contour = get_random_points(contour, n=32)  # retain n points from the contour
    # contour = get_evenly_spaced_points_by_distance(contour, n=3)  # retain n points from the contour
    # contour = get_evenly_spaced_points(contour, n=32)  # retain n points from the contour
    print(f'len contour = {len(contour)}')

    # Extract x and y coordinates from the contour positions
    x = contour[0]
    y = contour[1]

    coordinate = np.transpose(np.array([x, y, np.zeros(x.shape[0])]))

    new_contour = np.transpose(ffd(coordinate/640))

    new_contour *= 640

    x_fine, y_fine = new_contour[0], new_contour[1]

    shape = process_mask(new_contour, (640, 640))


    # # Plot the parametric spline on top of the background with transparency
    plt.scatter(*ffd.control_points()[:, [0, 1]].T * 640)
    # plt.scatter(x_fine, y_fine, linewidth=2, alpha=0.8)

    plt.imshow(shape, cmap='viridis', alpha=0.5)




    print(ffd)
    plt.title('FFD')
    plt.axis('off')
    plt.show()

# plot_parametric_splines(contour_positions, contour2)
plot_parametric_splines(mask_position, img)


