import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.ndimage import binary_closing
from cleanrl.custom_env import MedicalImageSegmentationEnv

# for _ in range(10):
n_control_points = 4
try:
    env = MedicalImageSegmentationEnv('/Users/brownie/GitHub/RL_dqn/synthetic_ds/synthetic_dataset.h5', n_control_points, 10, 0.5)
except:
    env = MedicalImageSegmentationEnv("C:\Users\Shadow\Documents\GitHub\RL_dqn\synthetic_ds\synthetic_dataset.h5", n_control_points, 10, 0.5)
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