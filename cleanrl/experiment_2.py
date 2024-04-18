from dqn import transform_array, MedicalImageSegmentationEnv
import random


# for _ in range(10):
Env = MedicalImageSegmentationEnv('utah_test_set.h5', 4, 10, 0.5)
action = [random.uniform(-1, 1) for _ in range(8)]
action2 = [0 for _ in range(8)]
observation, reward, done, info = Env.step(action)

