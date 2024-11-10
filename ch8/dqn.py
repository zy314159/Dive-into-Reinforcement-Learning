import numpy as np
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer

class QNet(nn.Module):
    def __init__(self,action_size):
        super(QNet,self).__init__()
        self.layers = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(action_size)
        )
    
    def forward(self,x):
        return self.layers(x)
    

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 1000
        self.batch_size = 32
        self.action_size = 2

        self.qnet = QNet(self.action_size)
        self.target_qnet = QNet(self.action_size)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.buffer = ReplayBuffer(self.buffer_size,self.batch_size)
    
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state,dtype=torch.float32).unsqueeze(0) #shape: (1,state_dim) 
            q_values = self.qnet(state) #shape: (1,action_size)
            return torch.argmax(q_values,dim=1).item()
        
    def sync_qnet(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def update(self,states,actions,rewards,next_states,dones):
        self.buffer.add(states,actions,rewards,next_states,dones)
        if len(self.buffer) < self.batch_size:
            return
        
        states,actions,rewards,next_states,dones = self.buffer.get_batch()
        states = torch.tensor(states,dtype=torch.float32)
        actions = torch.tensor(actions,dtype=torch.int64)
        rewards = torch.tensor(rewards,dtype=torch.float32)
        next_states = torch.tensor(next_states,dtype=torch.float32)
        dones = torch.tensor(dones,dtype=torch.int32)

        q_values = self.qnet(states)
        q = q_values[np.arange(self.batch_size),actions]

        next_q_values = self.target_qnet(next_states)
        next_q = torch.max(next_q_values,dim=1).values
        next_q.detach()
        target = rewards + (1-dones)*self.gamma*next_q

        loss = nn.functional.mse_loss(q,target)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    import gym

    episodes = 300
    sync_interval = 20
    agent = DQNAgent()
    env = gym.make('CartPole-v0')
    reward_history = []

    for e in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state,reward,done,_,info = env.step(action)
            agent.update(state,action,reward,next_state,done)
            state = next_state
            total_reward += reward
        
        if e % sync_interval == 0:
            agent.sync_qnet()
        
        reward_history.append(total_reward)

    import matplotlib.pyplot as plt

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(range(len(reward_history)),reward_history)
    plt.show()




    