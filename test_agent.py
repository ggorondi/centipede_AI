"""
Código para correr el agente (con el policy del modelo ya entrenado)
"""

import gymnasium as gym
from agent import GorondiAgent   
from state_extractor import StateExtractor 

agent = GorondiAgent()
extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')

obs, _ = env.reset()
total_reward = 0

while True:
    state = extractor.extract(obs)
    action = agent.action(state)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward+=reward

    if terminated or truncated:
        print(total_reward)
        break

print(agent.name())