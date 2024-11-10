from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)
    
    def add(self,state,action,reward,next_state,done):
        experience = (state,action,reward,next_state,done)
        self.buffer.append(experience)

    def get_batch(self):
        if self.batch_size > len(self.buffer):
            return None
        else:
            experiences = random.sample(self.buffer,self.batch_size) #shape: (batch_size,5)
            states = np.stack([exp[0] for exp in experiences],axis=0) #shape: (batch_size,state_dim)
            actions = np.array([exp[1] for exp in experiences]) #shape: (batch_size,)
            rewards = np.array([exp[2] for exp in experiences]) #shape: (batch_size,)
            next_states = np.stack([exp[3] for exp in experiences],axis=0)
            dones = np.array([exp[4] for exp in experiences]).astype(np.int32)

            return states,actions,rewards,next_states,dones
        
    def clear(self):
        self.buffer.clear()

if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    buffer = ReplayBuffer(buffer_size=1000,batch_size=32)

    for i in range(100):
        state = env.reset()[0]
        done = False

        while not done:
            action = env.action_space.sample()
            next_state,reward,done,_,info = env.step(action)
            buffer.add(state,action,reward,next_state,done)
            state = next_state
    
    states,actions,rewards,next_states,dones = buffer.get_batch()
    print(states.shape)
    print(actions.shape)
    print(rewards.shape)
    print(next_states.shape)
    print(dones.shape)
    buffer.clear()
