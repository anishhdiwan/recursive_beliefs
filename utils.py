import random
from secret_hitler_env import Agent

def init_draw_pile(num_liberal = 6, num_fascist = 11):
	'''
	Make binary array as a draw pile. Liberal policies are 0 and fascist policies are 1
	'''
	draw_pile = []
	for _ in range(num_liberal):
		draw_pile.append(0)
	for _ in range(num_fascist):
		draw_pile.append(1)

	random.shuffle(draw_pile)
	return draw_pile




def instantiate_agents():
	'''
	Generate agents in random order. Random order is important as the order in which agents take actions affects the optimal policy
	'''
	
	agent_indices = [0,1,2,3,4]
	agents = {}

	for i in range(3):
		# make three liberals 
		agent_idx = agent_indices.pop(random.randrange(len(agent_indices)))
		agents[agent_idx] = Agent("liberal", agent_idx)

	fascist_idx = agent_indices[0] # setting fascist to index 0 still maintians random order because of the random sampling beforehand
	hitler_idx = agent_indices[1]
	agents[fascist_idx] = Agent("fascist", fascist_idx, hitler_idx)
	agents[hitler_idx] = Agent("hitler", hitler_idx, fascist_idx)

	return agents, hitler_idx, fascist_idx