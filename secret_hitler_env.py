import random
from collections import namedtuple, deque
import numpy as np



class Agent:
	'''
	This is the agent class. It contains information that is known only to that specific instance of the agent. It also contains methods to take
	actions in the environment
	'''

	def __init__(self, agent_type, agent_idx, known_agent=None):

		'''
		agent_type = liberal, fascist, or hitler
		known_agent = index from 0 to 4. Type is inferred from the current agent's type
		'''
		assert (0 <= agent_idx <= 4), "agent index must be in [0,4]"
		agent_position = [0,0,0,0,0]
		self.agent_idx = agent_idx
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

	def get_legal_action_set(self, env):
		'''
		Look at the state and infer the set of legal actions that the agent can take. Same as the is_action_legal method from the env class
		This additionally checks to make sure that the agent's role allows that action. For example, only the president can kill other agents
		'''
		if env.state[0][1] == [0,0,0,0,0] and env.state[1][0] == 0:
			if self.agent_idx == env.president:
				# if there is neither a proposed nor an elected chancellor
				legal_actions = {"propose": [i for i in range(5) if i!=self.agent_idx]}
			else:
				legal_actions = {"propose": []}

		elif env.state[0][1] == [0,0,0,0,0] and env.state[1][0] == 1:
			# if there is no elected chancellor but there is a proposed one
			legal_actions = {"vote": [True, False]}

		elif env.state[0][1] != [0,0,0,0,0]:
			# if there currently is an elected chancellor then 
			# either the president draws three policies and discards one or the chancellor enacts one of the two policies given to them

			if (not env.policy_discarded) and (not env.policy_enacted):
				if self.agent_idx == env.president:
					legal_actions = {"discard_policy": [0,1]}
				else:
					legal_actions = {"discard_policy": []}
			elif env.policy_discarded and (not env.policy_enacted):
				if self.agent_idx == env.chancellor:
					legal_actions = {"enact_policy": [0,1]}
				else:
					legal_actions = {"enact_policy": []}
			elif env.policy_enacted and (env.state[2][1] in [4,5]):
				if self.agent_idx == env.president:
					legal_actions = {"kill": [i for i in range(5) if i!=self.agent_idx]}
				else:
					legal_actions = {"kill": []}


		return legal_actions




class SecretHitlerBoardGame:
	'''
	This is the environment class. It governs the state space, belief states, reward functions and transition model of the game as specified by 
	the game's rule set. The structure and methods of this class are modelled after the structure of most OpenAI gym RL environments
	'''

	def __init__(self, secret_hitler_idx=0, fascist_idx=1):
		self.observable_state = None
		self.done = False
		self.president = None
		self.secret_hitler_idx = secret_hitler_idx
		self.fascist_idx = fascist_idx


	def reset(self):
		'''
		The reset method resets the environment upon initialization. 
		'''

		self.done = False

		# features that affect the state		
		self.president = 0 # agent 0 is the first president
		self.chancellor = None # chancellor is undecided
		self.chancellor_is_proposed = False
		self.proposed_chancellor = None
		self.previous_govt = [None, None]
		self.enacted_lib_policies = 0 # no policies are enacted
		self.enacted_fas_policies = 0
		self.draw_pile_size = 17 # draw pile is full

		# features that affect state transitions and rewards
		self.votes = []
		self.vote_passed = None
		self.policy_discarded = None
		self.policy_enacted = None
		self.type_of_enacted_policy = None
		self.num_alive_agents = 5
		self.which_team_won = False
				
		self.state = None
		self.update_state()


	def step(self, action):
		'''
		The step function transitions the state of the environment to the next state. State transitions are absolute and not relative to any agent.
		This means that upon taking an action, the new state is one where the next agent takes an action. The current agent can hence not take 
		another action until all other agents have acted. This is done due to the turnwise nature of the game. 

		Note: during trajectory simulation (rollouts) within POMCP, an agent uses an estimated opponent policy 
		'''

		self.transition_state(action)

		next_observable_state = self.state
		reward = self.get_reward()
		done = self.done

		return next_observable_state, reward, done


	def transition_state(self, action):
		'''
		Change class variables and update self.state as per the current state of the game and the action that was played

		Actions are defined as key value pairs
		For example, an action could be {"kill": 3} to imply kill agent at index 3 or {"vote": False} to vote against a proposed chancellor
		'''

		action_is_legal, legal_actions = self.is_action_legal(action)
		# print(f"Action: {legal_actions} is {action_is_legal} legal")

		if action_is_legal:
			if list(action.keys())[0] == "propose":
				# chancellor is proposed to be the given index
				# pesident cannot propose themself as chancellor
				if action["propose"] == self.president:
					raise Exception("President can not propose themselves as the chancellor")
				self.chancellor_is_proposed = True
				# Reset the policy discard and enact state since a new round has started
				self.policy_discarded = False
				self.policy_enacted = False
				self.proposed_chancellor = action["propose"]

			elif list(action.keys())[0] == "vote":
				if len(self.votes) < self.num_alive_agents:
					# if all votes are yet to be given, repeat
					self.votes.append(action["vote"])
					print(f"Voting status: {self.votes}")
				if len(self.votes) == self.num_alive_agents:
					# on receiving all votes
					if self.votes.count(0) >= self.votes.count(1):
						print("Vote Failed")
						# if vote fails move on to the next president
						self.vote_passed = False
						self.chancellor_is_proposed = False
						self.proposed_chancellor = None
						self.president += 1
						if self.president >= 5:
							self.president = 0
					else:
						# if vote passes, set the chancellor and reset the proposal conditions
						print("Vote Passed")
						self.vote_passed = True
						self.chancellor = self.state[1][1].index(1)
						self.chancellor_is_proposed = False
						self.proposed_chancellor = None

					self.votes = []

			elif list(action.keys())[0] == "discard_policy":
				# reduce the size of the draw pile 
				self.policy_discarded = True
				self.draw_pile_size -= 3

			elif list(action.keys())[0] == "enact_policy":
				# enact the given policy and end the game if the enacted policies meet their desired count
				if action["enact_policy"] == 0:
					self.enacted_lib_policies += 1
					self.type_of_enacted_policy = 0
					if self.enacted_lib_policies == 5:
						self.which_team_won = 0
						self.done = True

				elif action["enact_policy"] == 1:
					self.enacted_fas_policies += 1
					self.type_of_enacted_policy = 1
					if self.enacted_fas_policies == 6:
						self.which_team_won = 1
						self.done = True

				self.policy_enacted = True

				if not self.enacted_fas_policies in [4,5]:
					# if the current count of enacted fascist policies is 4 or 5, the kill action is valid. In this case, don't change the state
					# else, reset chancellor, proposed chancellor, and proposal to default and increase president index
					self.president += 1
					if self.president >= 5:
						self.president = 0

					self.chancellor = None
					self.chancellor_is_proposed = False
					self.proposed_chancellor = None

			elif list(action.keys())[0] == "kill":
				# TO DO: the next president must be one that is alive and not the next one as per index
				# For now, an agent is also allowed to kill themself
				self.num_alive_agents -= 1
				if action["kill"] == self.secret_hitler_idx:
					self.done = True
					self.which_team_won = 0


			self.update_state()

		else:
			raise Exception("selected action is not valid, can not transition state")


	def get_reward(self):
		'''
		Return the reward depending on the state and the game interactions.
		The reward is an array of individual rewards for each agent

		REWARDS LIST
		on proposed vote passing
		+/- 1

		on enacting policy
		+/- 10

		on game ending by
		killing hiter or enacting required number of policies
		+/- 100

		'''
		reward = [0,0,0,0,0]

		if self.vote_passed:
			# the proposing president gets a reward of +1 on their vote being passed
			reward[self.state[0][0].index(1)] += 1

		if self.policy_enacted:
			if self.type_of_enacted_policy == 0:
				# on enacting a liberal policy, liberal agents get +10 reward, fascist agents get -10 reward
				for i in range(5):
					if i == self.secret_hitler_idx:
						reward[i] -= 10
					elif i == self.fascist_idx:
						reward[i] -= 10
					else:
						reward[i] += 10
			if self.type_of_enacted_policy == 1:
				# on enacting a fascist policy, liberal agents get -10 reward, fascist agents get +10 reward
				for i in range(5):
					if i == self.secret_hitler_idx:
						reward[i] += 10
					elif i == self.fascist_idx:
						reward[i] += 10
					else:
						reward[i] -= 10

		if self.done:
			if self.which_team_won == 0:
				# if liberals win, they get +100 reward and fascists get -100 reward
				for i in range(5):
					if i == self.secret_hitler_idx:
						reward[i] -= 100
					elif i == self.fascist_idx:
						reward[i] -= 100
					else:
						reward[i] += 100
			if self.which_team_won == 1:
				# if fascists win, they get +100 reward and liberals get -100 reward
				for i in range(5):
					if i == self.secret_hitler_idx:
						reward[i] += 100
					elif i == self.fascist_idx:
						reward[i] += 100
					else:
						reward[i] -= 100

		return reward			


	def is_action_legal(self, action):
		'''
		Is the current action legal given the current state. Sanity check method to ensure no invalid actions are taken
		'''

		if self.state[0][1] == [0,0,0,0,0] and self.state[1][0] == 0:
			# if there is neither a proposed nor an elected chancellor
			legal_actions = ["propose"]

		elif self.state[0][1] == [0,0,0,0,0] and self.state[1][0] == 1:
			# if there is no elected chancellor but there is a proposed one
			legal_actions = ["vote"]

		elif self.state[0][1] != [0,0,0,0,0]:
			# if there currently is an elected chancellor then 
			# either the president draws three policies and discards one or the chancellor enacts one of the two policies given to them

			if (not self.policy_discarded) and (not self.policy_enacted):
				legal_actions = ["discard_policy"]
			elif self.policy_discarded and (not self.policy_enacted):
				legal_actions = ["enact_policy"]
			elif self.policy_enacted and (self.state[2][1] in [4,5]):
				legal_actions = ["kill"]


		if list(action.keys())[0] in legal_actions:
			return True, legal_actions
		else:
			return False, "_"


	def update_state(self):
		'''
		Look at the current set of class variables and update self.state
		This is just a convenience method and does not actually transition state 

		State = 
		[
		current(proposed) govt = [[one hot vec of president], [one hot vec of chancellor]]
		proposed chancellor = [is_chancellor_proposed, [one hot vector of proposed chancellor]]
		board state = [num_liberal, num_fascist, num_policies_left]
		]
		'''

		temp = [
		[[0,0,0,0,0], [0,0,0,0,0]],
		[0, [0,0,0,0,0]],
		[0,0,0]
		]

		# if reset:
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

		self.state = temp

