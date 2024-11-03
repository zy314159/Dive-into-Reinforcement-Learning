import numpy as np

class NonStatBandit:
    def __init__(self,arms=10):
        self.arms=arms
        self.rates=np.random.rand(arms)
    
    def play(self,arm):
        rate=self.rates[arm]
        self.rates+=0.1*np.random.randn(self.arms)#add noise
        if np.random.rand()<rate:
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self,episilon,alpha,actions=10):
        self.episilon=episilon
        self.alpha=alpha
        self.Qs=np.zeros(actions)

    def update(self,action,reward):
        self.Qs[action]+=self.alpha*(reward-self.Qs[action])
    
    def get_action(self):
        if np.random.rand()<self.episilon:
            return np.random.randint(0,len(self.Qs))
        else:
            return np.argmax(self.Qs)
        
import matplotlib.pyplot as plt
alpha=0.8
epsilon=0.1
steps=1000
runs=200

all_rates=np.zeros((runs,steps))

for run in range(runs):
    bandit=NonStatBandit()
    agent=AlphaAgent(epsilon,alpha)
    total_reward=0
    rates=[] 

    for step in range(steps):
        action=agent.get_action()
        reward=bandit.play(action)
        agent.update(action,reward)
        total_reward+=reward
        rates.append(total_reward/(step+1))
    
    all_rates[run]=rates

mean_rewards=np.mean(all_rates,axis=0)
plt.plot(mean_rewards)
plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.show()