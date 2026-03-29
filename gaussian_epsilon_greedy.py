import numpy as np
import matplotlib.pyplot as plt
class BanditArm:
    def __init__(self,m):
        self.m=m
        self.m_estimate=0
        self.N=0
    def pull(self):
        return np.random.randn()+self.m
    def update(self,x):
        self.N+=1
        self.m_estimate=((self.N-1)*self.m_estimate+x)/self.N
def run_experiment(m1,m2,m3,eps,N):
    bandits=[BanditArm(m1),BanditArm(m2),BanditArm(m3)]
    means=np.array([m1,m2,m3])
    opt=np.argmax(means)
    count_suboptimal=0
    data=np.empty(N)
    for i in range(N):
        p=np.random.random()
        if p<eps:
            j=np.random.choice(len(bandits))
        else:
            j=np.argmax([b.m_estimate for b in bandits])
        x=bandits[j].pull()
        bandits[j].update(x)
        if j!=opt:
            count_suboptimal+=1
        data[i]=x

    cum_average=np.cumsum(data)/(np.arange(N)+1)
    plt.plot(cum_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    for b in bandits:
        print(b.m_estimate)

    print("eps",eps)
    print("percentage of suboptimal bandits:",count_suboptimal/N)
    return cum_average
if __name__=='__main__':
    m1,m2,m3=1.5,2.5,3.5
    c1=run_experiment(m1,m2,m3,.1,100000)
    c2 = run_experiment(m1, m2, m3, .05, 100000)
    c3 = run_experiment(m1, m2, m3, .01, 100000)
    plt.plot(c1,label='eps=0.1')
    plt.plot(c2,label='eps=0.05')
    plt.plot(c3,label='eps=0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    plt.plot(c1, label='eps=0.1')
    plt.plot(c2, label='eps=0.05')
    plt.plot(c3, label='eps=0.01')
    plt.legend()
    plt.show()
