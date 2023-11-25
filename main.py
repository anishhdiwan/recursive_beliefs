import game_env
import utils
import random
# seed = random.randrange(0,1000)
# print(seed)
random.seed(82)

agents, hitler_idx, fascist_idx = utils.instantiate_agents()
# print(agents)

env = game_env.SecretHitlerBoardGame(secret_hitler_idx=hitler_idx, fascist_idx=fascist_idx)
env.reset()
print(f"Secret hitler: {env.secret_hitler_idx}")
print(f"Fascist agent: {env.fascist_idx}")

for i in range(25):
	for j in range(5):
		legal_actions = agents[j].get_legal_action_set(env)
		print(f"Agent {j} legal actions: {legal_actions}")
		if legal_actions[list(legal_actions.keys())[0]] == []:
			print(f"Agent {j} has no legal actions. Skipping turn")
		else:
			action = {list(legal_actions.keys())[0]: random.choice(legal_actions[list(legal_actions.keys())[0]])}
			print(f"Action chosen: {action}")
			env.step(action)
			print(env.state)
			if len(env.dead_agents) > 0:
				print(f"Dead agents: {env.dead_agents}")

		print("-----")
	
	print("--------------------------")

# for i_episode in range(num_episodes):
# 	env.reset()
	
