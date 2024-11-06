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
        if np.random.rand() < self.episilon:
            return np.random.randint(0,len(self.Qs))#proability of random action
        else:
            return np.argmax(self.Qs)          #greedy action


runs=200
steps=1000
epsilons=[0.1,0.3,0.01]
means=[]

all_rates=np.zeros((runs,steps))

for epsilon in epsilons:
    for run in range(runs):
        bandit=Bandit()
        agent=Agent(epsilon)
        total_reward=0
        rates=[]

        for step in range(steps):
            action=agent.get_action()
            reward=bandit.play(action)
            agent.update(action,reward)
            total_reward+=reward
            rates.append(total_reward/(step+1))              

        all_rates[run]=rates

    mean_rates=np.mean(all_rates,axis=0)
    means.append(mean_rates)

import matplotlib.pyplot as plt
for i,mean in enumerate(means):
    plt.plot(mean,label='epsilon='+str(epsilons[i]))
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend(loc='best')
plt.show()

