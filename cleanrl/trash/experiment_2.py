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
action = env.action_space.sample()
print(action)
observation, reward, _, _, _ = env.step(action)
print(f'sum of observation = {np.sum(observation)}')
env.render()