import numpy as np
import matplotlib.pyplot as plt
n_trial=10000
epsilon=0.05
bandit_p=[.2,.5,.75]
class Bandit:
    def __init__(self,p):
        self.p=p #p is the winrate
        self.p_estimate=0.
        self.N=0. #num of samples
    def pull(self):
        return np.random.random() < self.p
    def update(self,x):
        self.N+=1
        self.p_estimate=(self.p_estimate*(self.N-1)+x)/self.N

def experiment():
    bandits=[Bandit(p) for p in bandit_p]
    rewards=np.zeros(n_trial)
    n_exploration=0
    n_exploitation=0
    n_opt=0
    optimal_i=np.argmax([b.p for b in bandits])
    print("optimal bandit:",optimal_i)
    for i in range(n_trial):
        if np.random.random()<epsilon:
            n_exploration+=1
            j = np.random.randint(len(bandit_p))
        else:
            n_exploitation+=1
            j=np.argmax([b.p_estimate for b in bandits])
        if j==optimal_i:
            n_opt+=1
        x=bandits[j].pull()
        rewards[i]=x
        bandits[j].update(x)
    for b in bandits:
        print("mean estimate",b.p_estimate)
    print("total reward",rewards.sum())
    print("overall winrate:",rewards.sum()/n_trial)
    print("time explored",n_exploration)
    print("time exploited", n_exploitation)
    print("times selected optimal bandit", n_opt)
    cum_rewards=np.cumsum(rewards)
    win_rates=cum_rewards/(np.arange(n_trial)+1)
    plt.plot(win_rates)
    plt.plot(np.ones(n_trial)*np.max(bandit_p))
    plt.show()
if __name__=="__main__":
    experiment()
