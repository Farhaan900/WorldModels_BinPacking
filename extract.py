'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from model import make_model
import env

MAX_FRAMES = 1000  # max length of carracing
MAX_TRIALS = 2000  # just use this to extract one trial.

render_mode = False  # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model(load_model=False)

# total_frames = 0
model.make_env()
for trial in range(MAX_TRIALS):  # 200 trials per worker
    try:
        random_generated_int = random.randint(0, 2 ** 31 - 1)
        filename = DIR_NAME + "/" + str(random_generated_int) + ".npz"
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        model.env.seed(random_generated_int)

        # random policy
        # model.init_random_model_params(stdev=np.random.rand()*0.01)

        model.reset()
        obs = model.env.reset()  # pixels
        done = False
        total_reward = 0
        # while not done:
        #     recording_obs.append(obs)
        #     print(obs['action_mask'])
        #     print([x for x in range(len(obs['action_mask'])) if x == 1])
        #
        #     action = env.get_action(obs)
        #     obs, reward, done, info = model.env.step(action)

        while not done:
            recording_obs.append(obs)
            print([x[0] for x in obs[0]])
            print(np.shape(obs))
            action = env.get_action(obs)
            obs, reward, done, info = model.env.step(action)
            print("reward", reward)
            total_reward += reward

            recording_action.append(action)

            # print("reward", reward)
            total_reward += reward

        # for frame in range(MAX_FRAMES):
        #   if render_mode:
        #     model.env.render("human")
        #   else:
        #     model.env.render("rgb_array")
        #
        #   recording_obs.append(obs)
        #
        #   z, mu, logvar = model.encode_obs(obs)
        #   action = model.get_action(z)
        #
        #   recording_action.append(action)
        #   obs, reward, done, info = model.env.step(action)
        #
        #   if done:
        #     break

        # total_frames += (frame+1)
        # print("dead at", frame+1, "total recorded frames for this worker", total_frames)
        print("total reward: ", total_reward)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)
    except gym.error.Error:
        print("stupid gym error, life goes on")
        model.env.close()
        model.make_env(render_mode=render_mode)
        continue
model.env.close()
