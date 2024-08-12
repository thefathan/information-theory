import numpy as np
import matplotlib.pyplot as plt
import os

# Variables
c_params = [0.2, 0.3, 0.4, 0.5] # parameter c
ivs = [0.623521, 0.623522] # initial values
l = 1000000 # length (N)
n = 60 # iteration
x = np.zeros((len(ivs), l)) # sequence for bernouli
y = np.zeros((len(ivs), l)) # sequence for logistic

def skew_bernouli_map(x, c): # Bernoulli function
    if x < c:
        return (x/c)
    else:
        return (x-c)/(1-c)

def logistic_map(x): # Logistic function
    return 4*x*(1-x)

def main(): 
    for c in c_params:
        for idx, iv in enumerate(ivs):
            x[idx, 0] = iv
            for i in range(1, x.shape[1]):
                x[idx, i] = skew_bernouli_map(x[idx, i-1], c)
        
        # 1-2 Bernoulli Skew Map
        plt.figure()
        plt.plot(x[0, :n+1], color='r', label=f"initial value = {x[0, 0]}", linewidth=1.25)
        plt.plot(x[1, :n+1], color='k', label=f"initial value = {x[1, 0]}", linewidth=1.25)
        plt.legend(loc='upper center', bbox_to_anchor=(0.79, 1.16))
        plt.xlim(0, n)
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1])
        plt.xlabel("n", fontsize=14)
        plt.ylabel("Xn", fontsize=14)
        plt.title(f'Bernouli Map; c = {c}', loc = "left")
        plt.savefig(f"result/1-2_bernouli_{c}.png")

        # 2-2 Bernoulli Invariant
        for idx, iv in enumerate(ivs):
            plt.figure()
            plt.hist(x[idx, :], bins=100, rwidth=0.4, color='r', density=True)
            plt.xlim(0, 1)
            plt.ylim(0, 2)
            plt.yticks([0, 0.5, 1, 1.5, 2])
            plt.hlines(1, 0, 1, color='b', linewidth=1)
            plt.xlabel("x", fontsize=14)
            plt.ylabel("invariant density", fontsize=14)
            plt.title(f'Bernouli invariant density; c={c}, initial value={iv}', loc = "left")
            plt.savefig(f"result/2-2_bernouli_{c}_{iv}.png")

    for idx, iv in enumerate(ivs):
        y[idx, 0] = iv
        for i in range(1, x.shape[1]):
            y[idx, i] = logistic_map(y[idx, i-1])

        # 1-1 Logistic Map
        plt.figure()
        plt.plot(y[0, :n+1], color='r', label=f"initial value = {y[0, 0]}", linewidth=1.25)
        plt.plot(y[1, :n+1], color='k', label=f"initial value = {y[1, 0]}", linewidth=1.25)
        plt.legend(loc='upper center', bbox_to_anchor=(0.79, 1.16))
        plt.xlim(0, n)
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1])
        plt.xlabel("n", fontsize=14)
        plt.ylabel("Xn", fontsize=14)
        plt.title(f'Logistic Map', loc = "left")
        plt.savefig(f"result/1-1_logistic.png")

        # 2-1 Logistic Invarant
        for idx, iv in enumerate(ivs):
            plt.figure()
            plt.hist(y[idx, :], bins=100, rwidth=0.4, color='r', density=True)
            plt.xlim(0, 1)
            plt.ylim(0, 2)
            plt.yticks([0, 0.5, 1, 1.5, 2])
            plt.hlines(1, 0, 1, color='b', linewidth=1)
            plt.xlabel("x", fontsize=14)
            plt.ylabel("invariant density", fontsize=14)
            plt.title(f'Logistic Map invariant density; initial value = {iv}', loc = "left")
            plt.savefig(f"result/2-1_logistic_{iv}.png")

if __name__ == "__main__":
    os.makedirs('result', exist_ok=True)
    main()