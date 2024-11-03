import numpy as np

class Bandit:
    def __init__(self,arms=10):
        self.rates=np.random.rand(arms)
    
    def play(self,arm):
        rate=self.rates[arm]
        if np.random.rand()<rate:
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self,episilon,action_size=10):
        self.episilon=episilon
        self.Qs=np.zeros(action_size)
        self.ns=np.zeros(action_size)
    
    def update(self,action,reward):
        self.ns[action]+=1
        self.Qs[action]+=(reward-self.Qs[action])/self.ns[action]

    def get_action(self):
        if np.random.rand()<self.episilon:
            return np.random.randint(0,len(self.Qs))#proability of random action
        else:
            return np.argmax(self.Qs)#greedy action

import matplotlib.pyplot as plt

steps=1000
epsilon=0.1

bandit=Bandit()
agent=Agent(epsilon)
total_reward=0
total_rewards=[]
rates=[]

for step in range(steps):
    action=agent.get_action()
    reward=bandit.play(action)
    agent.update(action,reward)
    total_reward+=reward

    total_rewards.append(total_reward)
    rates.append(total_reward/(step+1))

print("Total reward:",total_reward)
plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()