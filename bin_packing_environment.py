import gym
import numpy as np
from gym import spaces

"""
STATE:
Number of bags at each level
Item size

ACTION:
Choose bag
"""

BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10
import random
import os
from PIL import Image
import rendering
class BinPackingGymEnvironment(gym.Env):

    def __init__(self, env_config={}):
        self.record=False
        metadata = {  # Feel free to add anything useful here
        'render.modes': ['human', 'rgb_array'],
        }
        self.viewer = None
        self.should_render=True
        if self.record:
            DIR_NAME = 'record'
            if not os.path.exists(DIR_NAME):
                os.makedirs(DIR_NAME)
        
        config_defaults = {
            'bag_capacity': 9,#Or 99,
            'item_sizes': [2,3], #[1, 2, 3, 4, 5, 6, 7, 8, 9],
            'item_probabilities': [0.8,0.2],#[0, 0, 0, 1/3, 0, 0, 0, 0, 2/3],  # linear waste -> SS: -150 to -340
            # 'item_probabilities': [0.75, 0.25], # perfect pack -> SS: -20 to -100
            #'item_probabilities': [0.5, 0.5], #bounded waste ->  SS: -11 to -20
            'time_horizon': 1000,
        }

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val  # Creates variables like self.plot_boxes, self.save_files, etc
            if key not in env_config:
                env_config[key] = val
        print('Using bin size: ', self.bag_capacity)
        print('Using items sizes {} \nWith item probabilities {}'.format(self.item_sizes,self.item_probabilities))
        self.csv_file = '/opt/ml/output/intermediate/binpacking.csv'

        self.episode_count = 0

        # state: number of bags at each level, item size,
        self.observation_space = spaces.Box(low=np.array([0] * self.bag_capacity + [0]), high=np.array([self.time_horizon] * self.bag_capacity + [max(self.item_sizes)]))

        # actions: select a bag from the different levels possible
        self.action_space = spaces.Discrete(self.bag_capacity)
        self.env=self
   
    def _render(self, mode='human', close=False):
        pass

    def reset(self):
        
        
        if self.record:
            DIR_NAME = 'record'
            random_generated_int = random.randint(0, 2 ** 31 - 1)
            self.filename = DIR_NAME + "/" + str(random_generated_int) + ".npz"
            self.recording_obs = []
            self.recording_action = []
        

        self.time_remaining = self.time_horizon
        self.item_size = self.__get_item()
        self.num_full_bags = 0

        # an array of size bag capacity that keeps track of
        # number of bags at each level
        self.num_bins_levels = [0] * self.bag_capacity

        initial_state = self.num_bins_levels + [self.item_size]
        self.total_reward = 0
        self.waste = 0
        self.episode_count += 1
        self.bin_type_distribution_map = {}  # level to bin types, to the # of bins for each bin type.
        self.step_count = 0
        valid_actions = list()
        # get bin levels for which bins exist and item will fit
        for x in range(1, self.action_space.n):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        valid_actions.append(0)  # open new bag
        self.action_mask=[0 for x in range(self.action_space.n)]
        for i in valid_actions:
          self.action_mask[i]=1
        return initial_state

    def step(self, action):
        done = False
        self.step_count += 1
        if action >= self.bag_capacity:
            print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            reward = BIG_NEG_REWARD - self.waste
            done = True
        elif action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste = sum of empty spaces in all bags
            self.waste = self.bag_capacity - self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # can't insert item because bin of this level doesn't exist
            print('cannot insert item because bin of this level does not exist')
            reward = BIG_NEG_REWARD - self.waste
            done = True
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            # waste = empty space in the bag
            self.waste = -self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(action)
            if self.num_bins_levels[action] < 0:
                print(self.num_bins_levels[action])
            self.num_bins_levels[action] -= 1

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self.__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map

        valid_actions = list()
        # get bin levels for which bins exist and item will fit
        for x in range(1, self.action_space.n):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        valid_actions.append(0)  # open new bag
        self.action_mask=[0 for x in range(self.action_space.n)]
        for i in valid_actions:
          self.action_mask[i]=1
        return state, reward, done, info

    def __get_item(self):
        num_items = len(self.item_sizes)
        item_index = np.random.choice(num_items, p=self.item_probabilities)
        return self.item_sizes[item_index]

    def __update_bin_type_distribution_map(self, target_bin_util):
        if target_bin_util < 0 or target_bin_util + self.item_size > self.bag_capacity:
            print("Error: Invalid Bin Utilization/Item Size")
            return
        elif target_bin_util > 0 and target_bin_util not in self.bin_type_distribution_map:
            print("Error: bin_type_distribution_map does not contain " + str(target_bin_util) + " as key!")
            return
        elif target_bin_util > 0 and target_bin_util in self.bin_type_distribution_map and len(
                self.bin_type_distribution_map[target_bin_util]) == 0:
            print("Error: bin_type_distribution_map has no element at level " + str(target_bin_util) + " !")
            return
        elif target_bin_util == 0:  # opening a new bin
            if self.item_size not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[self.item_size] = {str(self.item_size): 1}
            elif str(self.item_size) not in self.bin_type_distribution_map[self.item_size]:
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] = 1
            else:
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] += 1
        else:
            key = np.random.choice(list(self.bin_type_distribution_map[target_bin_util].keys()))
            if self.bin_type_distribution_map[target_bin_util][key] <= 0:
                print("Error: Invalid bin count!")
                return
            elif self.bin_type_distribution_map[target_bin_util][key] == 1:
                del self.bin_type_distribution_map[target_bin_util][key]
            else:
                self.bin_type_distribution_map[target_bin_util][key] -= 1

            new_key = self.__update_key_for_bin_type_distribution_map(key, self.item_size)
            if (target_bin_util + self.item_size) not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[target_bin_util + self.item_size] = {new_key: 1}
            elif new_key not in self.bin_type_distribution_map[target_bin_util + self.item_size]:
                self.bin_type_distribution_map[target_bin_util + self.item_size][new_key] = 1
            else:
                self.bin_type_distribution_map[target_bin_util + self.item_size][new_key] += 1

    @staticmethod
    def __update_key_for_bin_type_distribution_map(key, item_size):
        parts = key.split(' ')
        parts.append(str(item_size))
        parts.sort()
        return " ".join(parts)

    def render(self, mode="human", close=False):
        self._render()


class BinPackingIncrementalWasteGymEnvironment(BinPackingGymEnvironment):

    def step(self, action):
        done = False
        if action >= self.bag_capacity:
            print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            reward = BIG_NEG_REWARD - self.waste
        elif action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste = sum of empty spaces in all bags 
            self.waste = self.bag_capacity - self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # can't insert item because bin of this level doesn't exist
            print('cannot insert item because bin of this level does not exist')
            reward = BIG_NEG_REWARD - self.waste
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.__update_bin_type_distribution_map(action)
            self.num_bins_levels[action] -= 1
            # waste = sum of empty spaces in all bags 
            self.waste = -self.item_size
            # reward is negative waste
            reward = -1 * self.waste

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self.__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        return state, reward, done, info


class BinPackingNearActionGymEnvironment(BinPackingGymEnvironment):

    def step(self, action):
        done = False
        invalid_action = not (self.__is_action_valid(action))
        if invalid_action:
            action = self.__get_nearest_valid_action(action)
            reward = BIG_NEG_REWARD
        else:
            reward = self.__insert_item(action)

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self._BinPackingGymEnvironment__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        return state, reward, done, info

    def __insert_item(self, action):
        if action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste added by putting in new item
            self.waste = self.bag_capacity - self.item_size
        else:  # insert in existing bag
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.num_bins_levels[action] -= 1
            # waste reduces as we insert item in existing bag 
            self.waste = -self.item_size
        # reward is negative waste
        reward = -1 * self.waste
        self._BinPackingGymEnvironment__update_bin_type_distribution_map(action)
        return reward

    def __get_nearest_valid_action(self, action):
        num_actions = self.bag_capacity
        # valid actions have
        valid_actions = list()
        for x in range(1, num_actions):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        if valid_actions:
            # get nearest valid action
            valid_action = min(valid_actions, key=lambda x: abs(x - action))
        else:
            valid_action = 0  # open new bag

        return valid_action

    def __is_action_valid(self, action):
        if action >= self.bag_capacity:
            print("Error: Invalid Action ", action)
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            print('cannot insert item because bin overflow')
            return False
        elif action == 0:  # new bag
            return True
        elif self.num_bins_levels[action] == 0:
            print('cannot insert item because bin of this level does not exist')
            return False
        else:  # insert in existing bag
            return True


class BinPackingContinuousActionEnv(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        # actions: select a bag from the different levels possible
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]),
                                       dtype=np.float32)

    def step(self, action):
        # clip to 0 and 1
        action = np.clip(action, 0, 1)
        # de-normalize to bin size and make it integer
        action = int(action * (self.bag_capacity - 1))
        return super().step(action)

class BinPacking2DMaskGymEnvironment(BinPackingGymEnvironment):#BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):

        env_config_forced = {
            "bag_capacity": 30,
            'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
            'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33],  # perfect pack
            #                  'item_probabilities': [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3], #linear waste
            'time_horizon': 15000,  # 10000
        }
        super().__init__(env_config_forced)


    def reset(self):
        state = super().reset()
        obs = self.bin_to_pic_encoder(state)
        if self.record:
            self.recording_obs.append(obs)
        return obs

    def step(self, action):
        if self.record:
            self.recording_action.append(action)
        state, rew, done, info = super().step(action)

        obs = self.bin_to_pic_encoder(state)
        if self.record:
            if not done:
                self.recording_obs.append(obs)
            if done:
                #print("We done here.")
                self.recording_obs = np.array(self.recording_obs, dtype=np.uint8)
                self.recording_action = np.array(self.recording_action, dtype=np.uint8)
                np.savez_compressed(self.filename, obs=self.recording_obs, action=self.recording_action)
        
        return obs, rew, done, state
    def _render(self, mode='human', close=False):
        if self.should_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                    return
            screen_width = 1200
            screen_height = 800
            if self.viewer is None:  # This only happens in the first run
                self.viewer = rendering.SimpleImageViewer(maxwidth=800)
            vis= self.bin_to_pic_encoder(self.num_bins_levels + [self.item_size])
            #vis = Image.fromarray(vis.astype('uint8'), 'RGB')
            #vis = vis.resize((1200,800), Image.ANTIALIAS)
#vis.save('image.png')
            #vis = rendering.Image('image.png', 0, 0)
            #self.viewer.add_geom(vis)
            return self.viewer.imshow(vis.astype('uint8'))#return_rgb_array=mode == 'rgb_array')
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def bin_to_pic_encoder(self, state):

        item_codes = [(0.9440215301806998, 0.9685659361783929, 0.9921261053440984), (0.9194156093810073, 0.9528181468665897, 0.9842522106881968), (0.8948096885813149, 0.9370703575547865, 0.9763783160322953), (0.8672664359861592, 0.9193540945790081, 0.967520184544406), (0.8436447520184545, 0.9036063052672049, 0.9596462898885044), (0.8200230680507498, 0.8878585159554018, 0.9517723952326028), (0.7964013840830451, 0.8721107266435986, 0.9438985005767013), (0.7653979238754326, 0.8541176470588235, 0.9333794694348327), (0.7260284505959247, 0.8373856209150327, 0.919600153787005), (0.6866589773164169, 0.8206535947712418, 0.9058208381391772), (0.647289504036909, 0.803921568627451, 0.8920415224913495), (0.5984313725490196, 0.7805305651672433, 0.8777854671280276), (0.548235294117647, 0.7529719338715878, 0.866958861976163), (0.4980392156862745, 0.7254133025759324, 0.8561322568242984), (0.447843137254902, 0.6978546712802769, 0.8453056516724337), (0.3969088811995388, 0.6668512110726644, 0.8303575547866207), (0.3565551710880431, 0.6392925797770088, 0.8146097654748174), (0.31620146097654767, 0.6117339484813534, 0.7988619761630142), (0.2758477508650519, 0.5841753171856978, 0.783114186851211), (0.23598615916955018, 0.5497116493656286, 0.7647058823529412), (0.2035063437139562, 0.5172318339100346, 0.7479738562091504), (0.17102652825836218, 0.4847520184544406, 0.7312418300653595), (0.13854671280276817, 0.4522722029988466, 0.7145098039215687), (0.10865051903114188, 0.41656286043829294, 0.689042675893887), (0.08404459823144944, 0.38506728181468663, 0.6644367550941945), (0.059438677431757014, 0.35357170319108033, 0.6398308342945022), (0.03483275663206459, 0.32207612456747403, 0.6152249134948098), (0.03137254901960784, 0.28567474048442904, 0.5642906574394464), (0.03137254901960784, 0.25319492502883506, 0.5160630526720492), (0.03137254901960784, 0.22071510957324106, 0.46783544790465204)]
        item_codes.reverse()
#[[100, 100, 0], [100, 0, 100], [100, 0, 0], [0, 100, 100], [0, 100, 0], [0, 0, 100], [200, 200, 0],
                     # [200, 0, 200], [200, 0, 0], [0, 200, 200], [0, 200, 0], [0, 0, 200]]
        max_color_num = 255
        part_size = 7
        encoding_multiplier = int(max_color_num / part_size)
        #print(encoding_multiplier)

        bins = state[:-1]

        encoded_img = np.zeros([64, 64, 3], dtype=np.uint8)
        for idx, x in enumerate(bins):
            lower_bound=60-(idx*2)
            start_pos=0
            while x >= 29:
                x-=15
                encoded_img[lower_bound-2][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[29])
                encoded_img[lower_bound-1][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[29])
                encoded_img[lower_bound][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[29])
                start_pos+=2
            if x>0:
                encoded_img[lower_bound-2][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[x])
                encoded_img[lower_bound-1][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[x])
                encoded_img[lower_bound][start_pos:start_pos+2]=tuple(int(255*a) for a in item_codes[x])
            #print("Down: "+str((x,idx,idy)))
        start_pos=6*state[len(state) - 1]
        encoded_img[60][start_pos:start_pos+6] = tuple(int(255*a) for a in item_codes[state[len(state) - 1]])
        encoded_img[61][start_pos:start_pos+6] = tuple(int(255*a) for a in item_codes[state[len(state) - 1]])
        encoded_img[62][start_pos:start_pos+6] = tuple(int(255*a) for a in item_codes[state[len(state) - 1]])
        encoded_img[63][start_pos:start_pos+6] = tuple(int(255*a) for a in item_codes[state[len(state) - 1]])
        #print(encoded_img)
        return encoded_img

class BinPackingActionMaskGymEnvironment(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        self.observation_space = spaces.Dict({
            # a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
            "action_mask": spaces.Box(
                0,
                1,
                shape=(self.action_space.n,)),
            "real_obs": self.observation_space
        })

    def reset(self):
        state = super().reset()
        valid_actions = self.__get_valid_actions()
        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]
        # obs=np.zeros([64, 64, 3], dtype=int) # old 2D version
        # obs[0][0][0] = np.array(state)
        mask = np.append(np.array(self.action_mask), 0)
        # print("mask len : ", len(mask))
        '''
        # old 2D version
        xid = 0
        for y in np.array(state):
            obs[0][xid] = y
            # print(xid)
            obs[1][xid] = mask[xid]
            xid += 1
        '''
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs

    def step(self, action):
        state, rew, done, info = super().step(action)
        valid_actions = self.__get_valid_actions()
        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]

        # obs = np.zeros([64, 64, 3], dtype=int)   # old 2D version
        # obs[0] = np.array(state)
        # obs[1] = np.array(self.action_mask)
        # obs = np.zeros([64, 64, 3])
        # obs[0][0][0] = np.array(state)
        mask = np.append(np.array(self.action_mask), 0)

        ''' old 2D version 
        xid = 0
        for y in np.array(state):
            obs[0][xid] = y
            obs[1][xid] = mask[xid]
            xid += 1
        '''
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, rew, done, info

    def __get_valid_actions(self):
        valid_actions = list()
        # get bin levels for which bins exist and item will fit
        for x in range(1, self.action_space.n):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        valid_actions.append(0)  # open new bag
        #print("Valid actions "+str(valid_actions))
        return valid_actions


if __name__ == '__main__':
    env_config = {
        "bag_capacity": 30,
        'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #'item_probabilities': [0, 0, 0, 1 / 3, 0, 0, 0, 0, 2 / 3],  # linear waste
        # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
        'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
        'time_horizon': 1000,
    }

    env = BinPacking2DMaskGymEnvironment(env_config)
    state = env.reset()
    done = False
    count = 0
    total_reward = 0
    while not done:
        text = input("Action: ") 
        action = int(text)
        state, reward, done, st = env.step(action)
        #for i in state:
        #    for j in i:
        #        print(j)
        
        #print(st)
        print(reward)
        env.render(close=False)
        #img = Image.fromarray(state.astype('uint8'), 'RGB')
        #img.show()
   
        total_reward += reward
        print("Action: {0}, Reward: {1:.1f}, Done: {2}"
              .format(action, reward, done))
    print("Total Reward: ", total_reward)
