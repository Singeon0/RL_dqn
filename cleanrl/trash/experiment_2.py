import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.ndimage import binary_closing
from cleanrl.custom_env import MedicalImageSegmentationEnv

# for _ in range(10):
n_control_points = 4
env = MedicalImageSegmentationEnv('/Users/brownie/GitHub/RL_dqn/synthetic_ds/synthetic_dataset.h5', n_control_points, 10, 0.5)
action = np.round(env.action_space.sample(), 2)
print(action)
env.render()
print(f'rew={env._compute_reward()}')
action = np.array(action)
observation, reward, _, _, _ = env.step(action)
print(f'reward after step = {reward}')
env.render()