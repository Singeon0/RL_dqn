import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.ndimage import binary_closing
from custom_env import process_mask


loaded_array = np.load('my_array.npy')
img = process_mask(loaded_array, (100, 100))
plt.imshow(img, cmap='gray')
plt.show()











# import matplotlib.pyplot as plt
#
# from custom_env import MedicalImageSegmentationEnv
# import random
#
# # for _ in range(10):
# n_control_points = 9
# env = MedicalImageSegmentationEnv('../synthetic_ds/synthetic_dataset.h5', n_control_points, 10, 0.5)
# action = [random.uniform(-1, 1) for _ in range(2*n_control_points)]
# action2 = [0 for _ in range(2*n_control_points)]
# # action3 is also a random action array but following a gaussian distribution center in 0 and std 1
# action3 = [random.gauss(0, 0.3) for _ in range(2*n_control_points)]
# observation, reward, done, info = env.step(action3)
# plt.imshow(observation, cmap='gray')
# plt.show()
# print(env.observation_space)
# print(env.action_space)
#
# # Generate a random sample within the box
# sample = env.action_space.sample()
#
# print(sample)

