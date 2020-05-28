import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control.rendering import SimpleImageViewer

import numpy as np
import os
import copy

NOOP = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

class PursuersEvaders(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_name = "mmap1.txt", catch_level = 2, terminal_reward = 10.0, ontarget_reward = 1.0, move_reward = 0.0, bump_reward = -0.2):
        self.viewer = SimpleImageViewer()
        self.n = None
        self.m = None
        self.catch_level = catch_level
        self.walls = []
        self.init_evaders = []
        self.init_pursuers = []
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(this_file_path, file_name) 
        with open(file_name, "r") as f:
            for i, row in enumerate(f):
                row = row.rstrip('\r\n')
                if self.n is not None and len(row) != self.n:
                    raise ValueError("Map's rows are not of the same dimension...")
                self.n = len(row)
                for j, col in enumerate(row):
                    if col == "P":
                        self.init_pursuers.append(self.n * i + j)
                    elif col == "E":
                        self.init_evaders.append(self.n * i + j)
                    elif col == "1":
                        self.walls.append(self.n * i + j)
            self.m = i + 1
        if self.m < 3 or self.n < 3:
            raise ValueError("Map too small...")
        if len(self.init_pursuers) < self.catch_level:
            raise ValueError("At least a sufficient number of pursuers needs to be specified...")
        if len(self.init_evaders) == 0:
            raise ValueError("At least one evaders needs to be specified...")
        self.evaders = copy.copy(self.init_evaders)
        self.pursuers = copy.copy(self.init_pursuers)
        self.n_states = self.n * self.m
        self.n_actions = 5 ** len(self.init_pursuers)
        self.terminal_reward = terminal_reward
        self.ontarget_reward = ontarget_reward
        self.move_reward = move_reward
        self.bump_reward = bump_reward
        self.action_space = spaces.Box(0, 4, (len(self.init_pursuers),))
        self.observation_space = spaces.Box(-1, 3, (3, 3))
        self.done = False
        
    def step(self, action):
        assert self.action_space.contains(action)
        if len(self.evaders) == 0:
            return self.build_observation(), 0.0, self.done, None
        else:
            new_state = self.take_action(action)
            reward = self.get_reward(new_state, action)
            self.pursuers = new_state
            self.take_evaders_action()
            return self.build_observation(), reward, self.done, None

    def reset(self):
        self.done = False
        self.evaders = copy.copy(self.init_evaders)
        self.pursuers = copy.copy(self.init_pursuers)
        return self.build_observation()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'human':
            grid = np.multiply(np.ones((self.n_states, 3), dtype = np.int8), np.array([0, 255, 0], dtype = np.int8))
            for e in self.evaders:
                grid[e] = np.array([255, 0, 0])
            for w in self.walls:
                grid[w] = np.array([0, 0, 0])
            for p in self.pursuers:
                grid[p] = np.array([0, 0, 255])
            grid = grid.reshape(self.m, self.n, 3)
            self.viewer.imshow(grid)
            return self.viewer.isopen
        elif mode == "rgb_array":
            return grid
        else:
            return

    def take_action(self, action):
        new_state = []
        for a, p in zip(action, self.pursuers):
            row = p // self.n
            col = p % self.n
            if a == DOWN and (row + 1) * self.n + col not in self.walls:
                row = min(row + 1, self.m - 1)
            elif a == UP and (row - 1) * self.n + col not in self.walls:
                row = max(0, row - 1)
            elif a == RIGHT and row * self.n + col + 1 not in self.walls:
                col = min(col + 1, self.n - 1)
            elif a == LEFT and row * self.n + col - 1 not in self.walls:
                col = max(0, col - 1)
            new_state.append(row * self.n + col)
        return new_state

    def take_evaders_action(self):
        new_goals = []
        for e in self.evaders:
            row = e // self.n
            col = e % self.n
            a = np.random.randn(0, 5)
            if a == DOWN and (row + 1) * self.n + col not in self.walls:
                row = min(row + 1, self.m - 1)
            elif a == UP and (row - 1) * self.n + col not in self.walls:
                row = max(0, row - 1)
            elif a == RIGHT and row * self.n + col + 1 not in self.walls:
                col = min(col + 1, self.n - 1)
            elif a == LEFT and row * self.n + col - 1 not in self.walls:
                col = max(0, col - 1)
            new_goals.append(row * self.n + col)
        self.evaders = new_goals

    def get_reward(self, new_state, action):
        reward = 0.0
        for i, p in enumerate(new_state):
            n = 1
            for x in new_state[i + 1:]:
                if x == p:
                    n += 1
            if n >= self.catch_level and p in self.evaders:
                reward += self.terminal_reward
                self.evaders.remove(p)
                if len(self.evaders) == 0:
                    self.done = True
            elif p in self.evaders:
                reward += self.ontarget_reward
            elif p == self.pursuers[i] and action[i] != NOOP:
                reward += self.bump_reward
            else:
                reward += self.move_reward
        return reward

    def build_observation(self):
        observations = []
        for p in self.pursuers:
            row = p // self.n
            col = p % self.n
            o = np.zeros((3, 3), dtype = np.int8)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if row + i < 0 or row + i >= self.m or col + j < 0 or col + j >= self.n:
                        o[i + 1][j + 1] = -1
                    else:
                        q = (row + i) * self.n + col + j
                        if q in self.walls:
                            o[i + 1][j + 1] = -1
                        elif q in self.evaders:
                            if q in self.pursuers:
                                o[i + 1][j + 1] = 3
                            else:
                                o[i + 1][j + 1] = 1
                        elif q in self.pursuers:
                            o[i + 1][j + 1] = 2
            o = o.tolist()
            for i, e in enumerate(o):
                o[i] = tuple(e)
            observations.append(tuple(o))
        return tuple(observations)
