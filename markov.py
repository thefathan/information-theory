import numpy as np
import matplotlib.pyplot as plt
import os

def create_parameters(p_1, p_2):
    t = p_2 / (p_1 + p_2)
    a = 1 / (1 - (p_1 + p_2))
    if a > 0:
        a_positive = True
        c1 = t * (1 - (1/a))
        c2 = t + ((1 - t)/a)
        a1 = 1 / c1
        a2 = 1 / (1 - c2)
    else:
        a_positive = False
        c1 = t + ((1 - t)/a)
        c2 = t * (1 - (1/a))
        a1 = 1 / c1
        a2 = 1 / (1 - c2)

    return t, a, a_positive, c1, c2, a1, a2

def threshold_function(x, t):
    return 0 if x < t else 1

def plm3(a_positive, x, a, a1, a2, c1, c2):
    if x < c1:
        return a1 * x
    elif c1 <= x < c2:
        if a_positive:
            return a * (x - c1)
        else:
            return a * (x - c2)
    elif c2 <= x <= 1:
        return a2 * (x - c2)
    else:
        return None  # Handle case when x is outside [0, 1]

def generate_sequence(x0, a_positive, a, a1, a2, c1, c2, l): # generate sequence using plm3 map
    x = np.zeros(l)
    x[0] = x0
    for i in range(1, l):
        x[i] = plm3(a_positive, x[i-1], a, a1, a2, c1, c2)
    return x

def plot(p_1, p_2, t, a_positive, a, a1, a2, c1, c2, x0, l):
    x = np.arange(0, 1.00000, 0.00001)
    y = np.array([plm3(a_positive, xi, a, a1, a2, c1, c2) for xi in x])
    
    # piecewise linear chaotic map 3
    x_ticks = [0, c1, t, c2, 1]
    x_labels = ['0', 'c1', 't', 'c2', '1']
    y_ticks = [0, t, 1]
    y_labels = ['0', 't', '1']
    
    c1_range = int(c1 / 0.00001) + 1
    c2_range = int(c2 / 0.00001) + 1

    plt.figure(figsize=(5, 5))
    plt.plot(x[:c1_range], y[:c1_range], color='k')
    plt.plot(x[c1_range:c2_range], y[c1_range:c2_range], color='b')
    plt.plot(x[c2_range:], y[c2_range:], color='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.vlines(t, 0, 1, color='k', linewidth=1, linestyles='dashed')
    plt.vlines(c2, 0, 1, color='k', linewidth=1, linestyles='dashed')
    plt.vlines(c1, 0, 1, color='k', linewidth=1, linestyles='dashed')
    plt.vlines(t, 0, t, color='k', linewidth=1, linestyles='dashed')
    plt.hlines(t, 0, 1, color='k', linewidth=1, linestyles='dashed')
    plt.title(f'Piecewise Linear Map; p1={p_1}, p2={p_2}', loc="left")
    plt.savefig(f"assignment3/1/MarkovMap_p1:{p_1}_p2:{p_2}.png")

    # generate sequence
    sequence = generate_sequence(x0, a_positive, a, a1, a2, c1, c2, l)

    # invariant density
    plt.figure()
    plt.hist(sequence, bins=100, rwidth=0.4, color='r', density=True)
    plt.xlim(0, 1)
    plt.ylim(0, 2)
    plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.hlines(1, 0, 1, color='b', linewidth=1)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("invariant density", fontsize=14)
    plt.title(f'Markov invariant density; p1={p_1}, p2={p_2}', loc="left")
    plt.savefig(f"assignment3/1/density_p1:{p_1}_p2:{p_2}.png")

def main():
    p_list = [(0.01, 0.1), (0.4, 0.2), (0.9, 0.3), (0.9, 0.9)]  # p list
    x0 = 0.51262323  # initial value
    l = 1000000  # length (N)

    for p in p_list:
        p_1, p_2 = p
        t, a, a_positive, c1, c2, a1, a2 = create_parameters(p_1, p_2)
        os.makedirs('assignment3/1', exist_ok=True)
        plot(p_1, p_2, t, a_positive, a, a1, a2, c1, c2, x0, l)

        c1_count = c00 = c01 = c10 = c11 = 0  # initialization of counters
        for i in range(l):
            b1 = threshold_function(x0, t)
            next_x = plm3(a_positive, x0, a, a1, a2, c1, c2)
            if next_x is None:  # Ensure valid value --> refer the last condition on plm3 function
                continue
            b2 = threshold_function(next_x, t)
            c1_count += b1  # number of 1
            c11 += b1 * b2
            c10 += b1 * (1 - b2)
            c01 += (1 - b1) * b2
            c00 += (1 - b1) * (1 - b2)
            x0 = next_x  # next mapping

        # calculate P
        p1 = c1_count / l
        p0 = 1 - p1
        p00 = c00 / l
        p01 = c01 / l
        p10 = c10 / l
        p11 = c11 / l
        p0_0 = p00 / p0  # P(S0|S0)
        p0_1 = p10 / p1  # P(S0|S1)
        p1_0 = p01 / p0  # P(S1|S0)
        p1_1 = p11 / p1  # P(S1|S1)

        # display - 5 values behind comma
        print(f'parameter p1: {p_1}, p2: {p_2} --> t: {t:.3f}, a: {a:.3f}, c1: {c1:.3f}, c2: {c2:.3f}, a1: {a1:.3f}, a2: {a2:.3f}')
        print(f'P(0): {p0:.5f}')
        print(f'P(1): {p1:.5f}')
        print(f'P(00): {p00:.5f}')
        print(f'P(01): {p01:.5f}')
        print(f'P(10): {p10:.5f}')
        print(f'P(11): {p11:.5f}')
        print(f'P(0|0): {p0_0:.5f}')
        print(f'P(0|1): {p0_1:.5f}')
        print(f'P(1|0): {p1_0:.5f}')
        print(f'P(1|1): {p1_1:.5f}')

def main2():
    p_list = [(0.01, 0.01), (0.05, 0.05), (0.1, 0.2), (0.1, 0.3), (0.2, 0.4), (0.2, 0.5), (0.3, 0.5), (0.4, 0.4), (0.4999, 0.5), (0.6, 0.7), (0.6, 0.8), (0.7, 0.8), (0.7, 0.9), (0.8, 0.9), (0.8, 0.95), (0.9, 0.95)]  # p list
    x0 = 0.51262323   # initial value
    l = 1000000  # length (N)

    for p in p_list:
        p_1, p_2 = p
        t, a, a_positive, c1, c2, a1, a2 = create_parameters(p_1, p_2)
        b_seq = ""
        for i in range(l):
            b1 = threshold_function(x0, t)
            b_seq += str(b1)
            x0 = plm3(a_positive, x0, a, a1, a2, c1, c2)

        # check result
        print(f"p1:{p_1}, p2:{p_2}", b_seq[:10])
        print("length", len(b_seq))
        # save
        os.makedirs(f'assignment3/2/p1:{p_1}, p2:{p_2}', exist_ok=True)
        with open(f'assignment3/2/p1:{p_1}, p2:{p_2}/p1:{p_1}, p2:{p_2}.txt', 'w') as f:
            f.write(b_seq)

if __name__ == "__main__":
    print('\n\nnumber 1\n')
    main()
    print('\n\nnumber 2\n')
    main2()
