import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

# Set the dimensions and number of images
image_size = (110, 110)
num_images = 1000

# Create empty arrays to store the images
data_images = np.zeros((num_images, image_size[0], image_size[1]), dtype=np.int64)
prediction_images = np.zeros((num_images, image_size[0], image_size[1]), dtype=np.int64)
groundtruth_images = np.zeros((num_images, image_size[0], image_size[1]), dtype=np.int64)

# Generate the images
for i in range(num_images):
    # Data images: Black background
    # data_images[i] = np.zeros(image_size, dtype=np.uint8)

    # Prediction images: Central circle occupying 10% of the image area
    circle_radius = int(np.sqrt(0.1 * image_size[0] * image_size[1] / np.pi))
    circle_center = (image_size[0] // 2, image_size[1] // 2)
    prediction_images[i] = cv2.circle(np.zeros(image_size, dtype=np.uint8), circle_center, circle_radius, 1, -1)

    # Ground truth images: Circle that is 10% larger than the prediction circle
    groundtruth_radius = int(circle_radius * 1.33)  # Increase the radius by 10%
    groundtruth_images[i] = cv2.circle(np.zeros(image_size, dtype=np.uint8), circle_center, groundtruth_radius, 1, -1)

    data_images[i] = groundtruth_images[i].astype(np.uint8) * 100 + np.random.normal(0, 3, image_size)
# Save the images to an HDF5 file
with h5py.File('synthetic_dataset.h5', 'w') as hdf:
    hdf.create_dataset('image', data=data_images)
    hdf.create_dataset('prediction', data=prediction_images)
    hdf.create_dataset('groundtruth', data=groundtruth_images)

# Plot a sample of the generated images
fig, axs = plt.subplots(3, 9, figsize=(20, 10))

for i in range(9):
    axs[0, i].imshow(data_images[i], cmap='gray')
    axs[0, i].axis('off')
    if i == 0:
        axs[0, i].set_title('Data Images')

    axs[1, i].imshow(prediction_images[i], cmap='gray')
    axs[1, i].axis('off')
    if i == 0:
        axs[1, i].set_title('Prediction Images')

    axs[2, i].imshow(groundtruth_images[i], cmap='gray')
    axs[2, i].axis('off')
    if i == 0:
        axs[2, i].set_title('Ground Truth Images')

plt.tight_layout()
plt.show()

# showing the first image of prediction and groundtruth
plt.imshow(prediction_images[0], cmap='gray')
plt.title('Prediction Image')
plt.axis('off')
plt.show()

plt.imshow(groundtruth_images[0], cmap='gray')
plt.title('Ground Truth Image')
plt.axis('off')
plt.show()