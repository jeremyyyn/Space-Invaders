# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch
import numpy as np
import time

class Env():
  def __init__(self, env, device):
    self.device = device
    self.window = 4 # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=4)
    self.training = False  # Consistent with model training mode
    self.screen = None
    self.lives = 3
    self.env = env
    self.done = None
    self.info = None

  def _get_state(self):
    state = cv2.resize(cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if True:
      # Reset internals
      self._reset_buffer()
      self.screen = self.env.reset()
      self.lives = 3

    # Perform up to 30 random no-ops before starting
    for _ in range(random.randrange(30)):
      self.screen, _, _, _ = self.env.step(0)  # Action 0 is no-op
      if self.done:
        self.screen = self.env.reset()
        
    # Process and return "initial" state
    observation = self._get_state()
    
    self.state_buffer.append(observation)

    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
          
    obs, reward, done, info = self.env.step(action)
    self.done = done
    self.screen = obs
    self.info = info

    obs = cv2.resize(cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY), (84, 84), interpolation=cv2.INTER_LINEAR)
    obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
      
    self.state_buffer.append(obs)

    if self.lives > self.info['lives']:
        reward -= 100
        self.lives = self.info['lives']
    
    return torch.stack(list(self.state_buffer), 0), reward, done, self.info

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return self.env.action_space.n

  def render(self):
    return self.env.render('rgb_array')

  def close(self):
    self.env.close()
