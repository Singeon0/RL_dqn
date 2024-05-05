import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.ndimage import binary_closing
from cleanrl.custom_env import MedicalImageSegmentationEnv

# for _ in range(10):
n_control_points = 4
env = MedicalImageSegmentationEnv('/Users/brownie/GitHub/RL_dqn/synthetic_ds/synthetic_dataset.h5', n_control_points, 10, 0.5, interval_action_space=0.15)
print(env.action_space.shape)
action = np.round(env.action_space.sample(), 2)
print(action)
env.render()
print(f'rew={env._compute_reward()}')
action = np.array(action)

array = np.zeros((2, 2, 2))

# Assign the values as per the provided code
# mu_x assignments
array[1, 1, 0] = 0.09
array[0, 0, 0] = -0.09
array[0, 1, 0] = -0.09
array[1, 0, 0] = 0.09

# mu_y assignments
array[1, 1, 1] = 0.09
array[0, 0, 1] = -0.09
array[0, 1, 1] = 0.09
array[1, 0, 1] = -0.09



observation, reward, _, _, _ = env.step(array)

print(f'reward after step = {reward}')
env.render()

import numpy as np

# Initialize a list to store the action samples
action_samples = []

# Sample the action 1000 times
for _ in range(1000):
    action_sample = env.action_sample(percentage=0.00, interval_action_space=0.15)
    action_samples.append(action_sample)

# Convert the list to a numpy array for easier calculations
action_samples = np.array(action_samples)

# Calculate and print the mean value of the actions
mean_action = np.mean(action_samples)
print(f"Mean value of actions: {mean_action}")

# Calculate and print the standard deviation of the actions
std_action = np.std(action_samples)
print(f"Standard deviation of actions: {std_action}")

