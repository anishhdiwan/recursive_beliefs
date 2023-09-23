import random
from collections import namedtuple, deque
import numpy as np


class SecretHitlerBoardGame:
	'''
	This is the environment class. It governs the state space, belief states, reward functions and transition model of the game as specified by 
	the game's rule set. The structure and methods of this class are modelled after the structure of most OpenAI gym RL environments
	'''

	def __init_(self):
		self.observable_state = None
		self.done = False


	def reset(self):
		'''
		The reset method resets the environment upon initialization. 
		'''

		self.done = False
		self.state = some_state_vector

	def step(self, action):
		'''
		The step function transitions the state of the environment to the next state. State transitions are absolute and not relative to any agent.
		This means that upon taking an action, the new state is one where the next agent takes an action. The current agent can hence not take 
		an action until all other agents have acted. This is done due to the turnwise nature of the game. 

		Note: during trajectory simulation (rollouts) within POMCP, an agent uses an estimated opponent policy 
		'''

		if not self.done:
			next_observable_state = self.get_next_state()
			reward = self.get_reward()
			self.done = self.update_done()

		return next_observable_state, reward, done


