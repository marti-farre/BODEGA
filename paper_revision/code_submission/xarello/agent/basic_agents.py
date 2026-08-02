class Agent():
    def choose_action(self, observation):
        return None
    
    def learn(self, observation_old, action, observation_new, reward, terminated, truncated):
        pass

class RandomAgent(Agent):
    def __init__(self, env):
        self.action_space = env.action_space
        
    def choose_action(self, observation):
        return self.action_space.sample()
    
    def learn(self, observation_old, action, observation_new, reward, terminated, truncated):
        pass