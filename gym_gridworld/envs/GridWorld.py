import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
import os

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_name = "map1.txt", fail_rate = 0.0, terminal_reward = 1.0, move_reward = 0.0, bump_reward = -0.5, bomb_reward = -1.0):
        self.viewer = None
        self.n = None
        self.m = None
        self.bombs = []
        self.walls = []
        self.goals = []
        self.start = None
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(this_file_path, file_name) 
        with open(file_name, "r") as f:
            for i, row in enumerate(f):
                row = row.rstrip('\r\n')
                if self.n is not None and len(row) != self.n:
                    raise ValueError("Map's rows are not of the same dimension...")
                self.n = len(row)
                for j, col in enumerate(row):
                    if col == "x" and self.start is None:
                        self.start = self.n * i + j
                    elif col == "x" and self.start is not None:
                        raise ValueError("There is more than one starting position in the map...")
                    elif col == "G":
                        self.goals.append(self.n * i + j)
                    elif col == "B":
                        self.bombs.append(self.n * i + j)
                    elif col == "1":
                        self.walls.append(self.n * i + j)
            self.m = i + 1
        if len(self.goals) == 0:
            raise ValueError("At least one goal needs to be specified...")
        self.n_states = self.n * self.m
        self.n_actions = 4
        self.fail_rate = fail_rate
        self.state = self.start
        self.terminal_reward = terminal_reward
        self.move_reward = move_reward
        self.bump_reward = bump_reward
        self.bomb_reward = bomb_reward
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.done = False
        
    def _step(self, action):
        assert self.action_space.contains(action)
        if self.state in self.goals or np.random.rand() < self.fail_rate:
            return self.state, 0.0, self.done, None
        else:
            new_state = self._take_action(action)
            reward = self._get_reward(new_state)
            self.state = new_state
            return self.state, reward, self.done, None

    def _reset(self):
        self.done = False
        self.state = self.start
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'human':
            grid = np.multiply(np.ones((self.n_states, 3), dtype = np.int8), np.array([0, 255, 0], dtype = np.int8))
            for g in self.goals:
                grid[g] = np.array([255, 0, 0])
            for b in self.bombs:
                grid[b] = np.array([255, 255, 0])
            for w in self.walls:
                grid[w] = np.array([0, 0, 0])
            grid[self.state] = np.array([0, 0, 255])
            grid = grid.reshape(self.m, self.n, 3)
            self.viewer.imshow(grid)
            return self.viewer.isopen
        elif mode == "rgb_array":
            return grid
        else:
            return      

    def _take_action(self, action):
        row = self.state / self.n
        col = self.state % self.n
        if action == DOWN and (row + 1) * self.n + col not in self.walls:
            row = min(row + 1, self.m - 1)
        elif action == UP and (row - 1) * self.n + col not in self.walls:
            row = max(0, row - 1)
        elif action == RIGHT and row * self.n + col + 1 not in self.walls:
            col = min(col + 1, self.n - 1)
        elif action == LEFT and row * self.n + col - 1 not in self.walls:
            col = max(0, col - 1)
        new_state = row * self.n + col
        return new_state

    def _get_reward(self, new_state):
        if new_state in self.goals:
            self.done = True
            return self.terminal_reward
        elif new_state in self.bombs:
            return self.bomb_reward
        elif new_state == self.state:
            return self.bump_reward
        return self.move_reward