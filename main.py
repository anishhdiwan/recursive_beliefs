import secret_hitler_env
import random


env = secret_hitler_env.SecretHitlerBoardGame()
agents = secret_hitler_env.instantiate_agents()

for i_episode in range(num_episodes):
	obs = env.reset()
	
