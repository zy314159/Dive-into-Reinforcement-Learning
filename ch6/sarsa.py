import sys
import os
sys.path.append(os.getcwd())

from collections import defaultdict,deque
import numpy as np
from common.utils import greedy_probs

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.action_size = 4
        self.epsilon = 0.1
    
        random_actions = {i:0.25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)# store the last two transitions

    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions,p=probs)
        return action
    
    def reset(self):
        self.memory.clear()

    def update(self,state,action,reward,done):
        self.memory.append((state,action,reward,done))
        if len(self.memory) < 2:
            return
        
        state,action,reward,done = self.memory[0]
        next_state,next_action,next_reward,next_done = self.memory[1]
        
        next_q = 0 if done else self.Q[next_state,next_action]

        target = reward + self.gamma * next_q
        self.Q[state,action] += self.alpha * (target - self.Q[state,action])

        self.pi[state] = greedy_probs(self.Q,state,self.epsilon,self.action_size)


if __name__ == '__main__':
    from common.gridworld import GridWorld

    env = GridWorld()
    agent = SarsaAgent()

    episodes = 100000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward,done = env.step(action)

            agent.update(state,action,reward,done)

            if done:
                agent.update(next_state,0,0,True)
                break
            state = next_state

    env.render_q(agent.Q)