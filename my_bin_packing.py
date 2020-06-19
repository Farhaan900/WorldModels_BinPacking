from bin_packing_environment import BinPackingGymEnvironment, BinPackingActionMaskGymEnvironment
import csv
import random
import numpy as np


def get_action(state):
    # state = state['real_obs']
    # num_bins_level = state[:-1]
    # item_size = state['real_obs'][-1]
    # bag_capacity = len(state['real_obs']) - 1

    # print(state)
    # if item_size == 0:
    #     print('item size should be larger than 0')
    #     return 0

    # if item_size == bag_capacity:
    #     return 0  # new bag

    return [0]

    random_action = random.choice([x for x in range(len(state[1])) if state[1][x][0] == 1])
    return [random_action]

    return [0]

def get_human_action():

    action = int(input())

    print(action)

    return [action]

env_config = {
    "bag_capacity": 63,
    'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
    'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
    #                  'item_probabilities': [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3], #linear waste
    'time_horizon': 1000,  # 10000
}

env = BinPackingActionMaskGymEnvironment(env_config)
state = env.reset()

done = False
total_reward = 0
while not done:
    print([x[0] for x in state[0]])
    print(np.shape(state))
    # action = get_action(state)
    action = get_human_action()
    print(action)
    state, reward, done, info = env.step(action)
    print("reward", reward)
    total_reward += reward

print("Total reward for best fit baseline agent: ", total_reward)


