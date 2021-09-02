#inputQueue is presented as [[task0req],[task1req],....,[taskNreq]] for n submited tasks
#each taskIreq is presented as [rCPU,rRAM,eTime,dTime] (requiredCPU,requiredRAM,executionTime,deadlineTime)
#inputQueue = [[rCPU1,rRAM1,eTime1,dTime1],[rCPU2,rRAM2,eTime2,dTime2],....,[rCPUn,rRAMn,eTimen,dTime]] for n submitted tasks
#inputQueue CPU index of task i: inputQueue[i][0]
#inputQueue RAM index of task i: inputQueue[i][1]
#inputQueue eTime index of task i: inputQueue[i][2]
#inputQueue dTime index of task i: inputQueue[i][3]
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_CPU = 10
MAX_RAM = 4096

class SchedulerEnv(gym.Env):
    
    def __init__(self,inputQueue):
        super(SchedulerEnv, self).__init__()

        self.inputQueue = inputQueue
        obs = {
            'cpu':gym.spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32),
            'ram':gym.spaces.Box(low=0, high=1000, shape=(10,), dtype=np.float32),
            'scheduledNode':gym.spaces.Box(low=1, high=10, shape=(100,), dtype=np.int32),
            'scheduledRTime':gym.spaces.Box(low=0, high=20, shape=(100,), dtype=np.int32)
        }
        self.action_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([100,10]), dtype=np.int32)
        self.observation_space = gym.spaces.Dict(obs)

    def step(self, action):
        # Execute one time step within the environment
        # self._take_action(action)

        chosen_task = action[0]
        chosen_node = action[1]

        if chosen_task == 0 or chosen_node == 0:

        else:
            if MAX_CPU-self.cpu[chosen_node]-self.inputQueue[chosen_task][0]
        # self.current_step += 1

        # if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
        #     self.current_step = 0

        # delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cpu = np.zeros((10,), dtype=np.float32)
        self.ram = np.zeros((10,), dtype=np.float32)
        self.scheduledNode = np.zeros((100,), dtype=int)
        self.scheduledRTime = np.zeros((100,), dtype=int)
        initialState = {
            'cpu':self.cpu,
            'ram':self.ram,
            'scheduledNode':self.scheduledNode,
            'scheduledRTime':self.scheduledRTime     
        }
        return initialState

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')