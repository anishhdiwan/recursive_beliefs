import random
from collections import namedtuple, deque
import numpy as np



class Agent:
	'''
	This is the agent class. It contains information that is known only to that specific instance of the agent. It also contains methods to take
	actions in the environment
	'''

	def __init__(self, agent_type, agent_idx,  known_agent=None):

		'''
		agent_type = liberal, fascist, or hitler
		known_agent = index from 0 to 4. Type is inferred from the current agent's type
		'''
		assert (0 <= agent_idx <= 4), "agent index must be in [0,4]"
		agent_position = [0,0,0,0,0]
		self.agent_position = agent_position[agent_idx]
		self.agent_type = agent_type

		if agent_type == "liberal":
			self.agent_role = [0,0]
			self.known_role = [[0,0,0,0,0], [0,0]]

		elif agent_type == "fascist":
			assert (known_agent is not None) and (0 <= known_agent <= 4), "please provide the index of the secret hitler or ensure that it is in [0,4]"
			self.agent_role = [1,0]
			known_role = [[0,0,0,0,0], [0,0]]
			known_role[0][known_agent] = 1
			known_role[1] = [1,1]
			self.known_role = known_role

		elif agent_type == "hitler":
			assert (known_agent is not None) and (0 <= known_agent <= 4), "please provide the index of the other fascist or ensure that it is in [0,4]"
			self.agent_role = [1,1]
			known_role = [[0,0,0,0,0], [0,0]]
			known_role[0][known_agent] = 1
			known_role[1] = [1,0]
			self.known_role = known_role


		self.drawn_policies = None # the set of policies drawn from the draw pile if the agent is the president. Else, this is none
		self.policies_from_president = None # the set of policies handed over by the president if the agent is the chancellor. Else, this is none

	def get_legal_action_set():
		'''
		Look at the state and infer the set of legal actions that the agent can take
		'''


class SecretHitlerBoardGame:
	'''
	This is the environment class. It governs the state space, belief states, reward functions and transition model of the game as specified by 
	the game's rule set. The structure and methods of this class are modelled after the structure of most OpenAI gym RL environments
	'''

	def __init_(self):
		self.observable_state = None
		self.done = False
		self.president = None


	def reset(self):
		'''
		The reset method resets the environment upon initialization. 
		'''

		self.done = False
		
		self.president = 0 # agent 0 is the first president
		self.chancellor = None # chancellor is undecided
		self.chancellor_is_proposed = False
		self.proposed_chancellor = None
		self.previous_govt = [None, None]
		self.enacted_lib_policies = 0 # no policies are enacted
		self.enacted_fas_policies = 0
		self.draw_pile_size = 17 # draw pile is full
				
		self.state = self.update_state(reset = True)


	def step(self, action):
		'''
		The step function transitions the state of the environment to the next state. State transitions are absolute and not relative to any agent.
		This means that upon taking an action, the new state is one where the next agent takes an action. The current agent can hence not take 
		an action until all other agents have acted. This is done due to the turnwise nature of the game. 

		Note: during trajectory simulation (rollouts) within POMCP, an agent uses an estimated opponent policy 
		'''

		legal_action = self.is_legal_action(action)

		if legal_action and not self.done:
			next_observable_state = self.update_state()
			reward = self.get_reward()
			self.done = self.update_done()

		return next_observable_state, reward, done


	def update_state(self):
		'''
		State = 
		[
		current(proposed) govt = [[one hot vec of president], [one hot vec of chancellor]]
		proposed chancellor = [is_chancellor_proposed, [one hot vector of proposed chancellor]]
		board state = [num_liberal, num_fascist, num_policies_left]
		]
		'''

		temp = [
		[[0,0,0,0,0], [0,0,0,0,0]],
		[0, [0,0,0,0,0]]
		[0,0,0]
		]

		if reset:
			if self.president != None:
				temp[0][0][self.president] = 1

			if self.chancellor != None:
				temp[0][1][self.chancellor] = 1

			if self.chancellor_is_proposed:
				assert self.proposed_chancellor != None
				temp[1][0] = 1
				temp[1][1][self.proposed_chancellor] = 1

			temp[2][0] = self.enacted_lib_policies
			temp[2][1] = self.enacted_fas_policies
			temp[2][2] = self.draw_pile_size

