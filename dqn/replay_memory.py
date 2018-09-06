"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    
    
    self.states = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.states_plus_1 = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.states_plus_2 = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)


  def add(self, screen, reward, action, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def sample_g(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'states', 'states_plus_1', 'states_plus_2'],
            [self.actions, self.rewards, self.screens, self.terminals, self.states, self.states_plus_1, self.states_plus_2])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'states', 'states_plus_1', 'states_plus_2'],
            [self.actions, self.rewards, self.screens, self.terminals, self.states, self.states_plus_1, self.states_plus_2])):
      array = load_npy(os.path.join(self.model_dir, name))
      
  def sample(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    indexes_plus_1 = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.states[len(indexes), ...] = self.getState(index - 2)
      self.states_plus_1[len(indexes), ...] = self.getState(index - 1)
      self.states_plus_2[len(indexes), ...] = self.getState(index)
      indexes.append(index -1)
      indexes_plus_1.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    actions_plus_1 = self.actions[indexes_plus_1]
    rewards_plus_1 = self.rewards[indexes_plus_1]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.states, (0, 2, 3, 1)), actions, rewards, np.transpose(self.states_plus_1, (0, 2, 3, 1)), actions_plus_1, \
        rewards_plus_1, np.transpose(self.states_plus_2, (0, 2, 3, 1)), terminals
    else:
      return self.states, actions, rewards, self.states_plus_1, actions_plus_1, rewards_plus_1, self.states_plus_2, terminals
