from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self):
       self.gamma = 0.9
       self.alpha = 0.8
       self.epsilon = 0.1
       self.action_size = 4
       self.Q = defaultdict(lambda: 0)

    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.action_size))
        else:
            qs = [self.Q[state,a] for a in range(self.action_size)]
            return np.argmax(qs)
        
    def update(self,state,action,reward,next_state,done):
        next_q = 0 if done else max([self.Q[next_state,a] for a in range(self.action_size)])
        target = reward + self.gamma * next_q
        error = target - self.Q[state,action]
        self.Q[state,action] += self.alpha * error

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    from common.gridworld import GridWorld

    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward,done = env.step(action)
            agent.update(state,action,reward,next_state,done)

            if done:
                break

            state = next_state

    env.render_q(agent.Q)