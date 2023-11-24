import secret_hitler_env
import utils
import random


agents, hitler_idx, fascist_idx = utils.instantiate_agents()
print(agents)

env = secret_hitler_env.SecretHitlerBoardGame(secret_hitler_idx=hitler_idx, fascist_idx=fascist_idx)

print(env.secret_hitler_idx)
print(env.fascist_idx)

env.reset()

print(env.state)

agent0_actions = agents[0].get_legal_action_set(env)

print(agent0_actions)

test_action = {"propose": 1}
env.step(test_action)
print(env.state)

agent1_actions = agents[1].get_legal_action_set(env)

print(agent1_actions)

test_action = {"vote": True}
env.step(test_action)
print(env.state)

agent2_actions = agents[2].get_legal_action_set(env)

print(agent2_actions)

# for i_episode in range(num_episodes):
# 	env.reset()
	
