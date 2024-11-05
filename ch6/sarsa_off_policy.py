import sys
import os

sys.path.append(os.getcwd())
from collections import defaultdict,deque
import numpy as np
from common.utils import greedy_probs


class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {a:0.25 for a in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0.0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions, p=probs)
        return action

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward,done):
        self.memory.append((state,action,reward,done))
        if len(self.memory) < 2:
            return
        
        state,action,reward,done = self.memory[0]
        next_state,next_action,next_reward,next_done = self.memory[1]

        if done:
             next_q = 0
             rho =1
        else:
            next_q = self.Q[next_state,next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho * (reward + self.gamma * next_q)
        self.Q[state,action] += self.alpha * (target - self.Q[state,action])

        self.pi[state] = greedy_probs(self.Q,state,0)
        self.b[state] = greedy_probs(self.Q,state,self.epsilon)

if __name__ == '__main__':
    from common.gridworld import GridWorld
    env = GridWorld()
    agent = SarsaOffPolicyAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        done=False

        while True:
            action = agent.get_action(state)
            next_state,reward,done = env.step(action)
            agent.update(state,action,reward,done)

            if done:
                agent.update(next_state,0,0,done)
                break
            state = next_state

    env.render_q(agent.Q)