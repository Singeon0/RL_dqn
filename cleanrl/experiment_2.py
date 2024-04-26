import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.ndimage import binary_closing
from custom_env import process_mask


# loaded_array = np.load('my_array.npy')
# img = process_mask(loaded_array, (100, 100))
# plt.imshow(img, cmap='gray')
# plt.show()

import matplotlib.pyplot as plt

from custom_env import MedicalImageSegmentationEnv
import random

# for _ in range(10):
n_control_points = 16
env = MedicalImageSegmentationEnv('../synthetic_ds/synthetic_dataset.h5', n_control_points, 10, 0.5)
action = [random.uniform(-0.5, 0.5) for _ in range(2*n_control_points)]
action2 = [0 for _ in range(2*n_control_points)]
# action3 is also a random action array but following a gaussian distribution center in 0 and std 1
# Define the shape of the action space
shape = (int(np.sqrt(n_control_points)), int(np.sqrt(n_control_points)), 2)
sample = np.random.uniform(low=-10, high=10, size=shape)
print(f'sample = {sample}')

# Generate a random sample from the action space
data = [[[-0.18129767,  0.0750408 ],
  [ 0.06394202, -0.15285999],
  [-0.18305524,  0.34976196]],

 [[ 0.07980525, -0.0373291 ],
  [ 0.46010008,  0.07791017],
  [-0.08197941, -0.4065484 ]],

 [[-0.0913413,  -0.01813539],
  [ 0.48836026,  0.1968941 ],
  [-0.0641389,   0.26740518]]]

# Convert list to numpy array
action3 = np.array(data)
# print(action3)

observation, reward, _, _, _ = env.step(sample)
print(f'sum of observation = {np.sum(observation)}')
plt.imshow(observation, cmap='gray')
plt.show()
