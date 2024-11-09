from collections import defaultdict
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from common.utils import greedy_probs

class QLearningAgent:
    def __init__(self):
        self.alpha = 0.8
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {i:0.25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    
    def reset(self):
        pass

    def update(self,state,action,reward,next_state,done):
        next_q = 0 if done else max([self.Q[next_state,a] for a in range(self.action_size)])

        target = reward + self.gamma * next_q
        error = target - self.Q[state,action]

        self.Q[state,action] += self.alpha * error

        self.pi[state] = greedy_probs(self.Q,state,self.epsilon,self.action_size)
        self.b[state] =greedy_probs(self.Q,state,0,self.action_size)

if __name__ == '__main__':
    from common.gridworld import GridWorld

    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward,done = env.step(action)
            agent.update(state,action,reward,next_state,done)

            if done:
                break

            state = next_state

    env.render_q(agent.Q)


