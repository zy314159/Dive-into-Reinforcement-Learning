from collections import defaultdict
import numpy as np


class TdAgent:
    def __init__(self):
        self.gamma = 0.9 # discount factor
        self.alpha = 0.1 # learning rate
        self.action_size = 4 # up, down, left, right

        random_actions = {i:0.25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
    
    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions,p=probs)
        return action
    
    def eval(self,state,reward,next_state,done):
        next_V =  0 if done else self.V[next_state]# The value function of thr target state is 0 if it is terminal state
        target = reward + self.gamma * next_V

        self.V[state] += (target - self.V[state]) * self.alpha

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    from common.gridworld import GridWorld

    env = GridWorld()
    agent = TdAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.get_action(state)
            next_state,reward,done = env.step(action)

            agent.eval(state,reward,next_state,done)
     
            if done:
                break
            state = next_state

    env.render_v(agent.V)
    
