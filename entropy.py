import numpy as np
import matplotlib.pyplot as plt
import os

# Variables
c_params = [0.3, 0.4] # parameter c
t_params = [0.3, 0.4] # parameter t
ivs = [0.752352] # initial values
l = 1000000 # length (N)
n = 60 # iteration

def skew_bernouli_map(x, c): # Bernoulli function
    if x < c:
        return (x/c)
    else:
        return (x-c)/(1-c)

def treshold_function(x, t): # threshold (make it binary (1 or 0))
    if x < t:
        return 0
    else:
        return 1

def main():
    for c in c_params:
        for x in ivs:
            for t in t_params:
                c1 = c00 = c01 = c10 = c11 = 0 # initialization of counters
                for i in range (l):
                    b1 = treshold_function(x, t)
                    b2 = treshold_function(skew_bernouli_map(x, c), t)
                    c1 += b1 # number of 1
                    c11 += b1 * b2
                    c10 += b1 * (1 - b2)
                    c01 += (1 - b1) * b2
                    c00 += (1 - b1) * (1 - b2) 
                    x = skew_bernouli_map(x, c) # next mapping

                # calculate P
                p1 = c1 / l
                p0 = 1 - p1
                p00 = c00 / l
                p01 = c01 / l
                p10 = c10 / l
                p11 = c11 / l
                p0_0 = p00 / p0 #P(S0|S0) 
                p0_1 = p10 / p1 #P(S0|S1) 
                p1_0 = p01 / p0 #P(S1|S0) 
                p1_1 = p11 / p1 #P(S1|S1)

                # display
                print('parameter c:', c, '\nthreshold t:', t) 
                print(f'P(0): {p0:.3f}')
                print(f'P(1): {p1:.3f}')
                print(f'P(00): {p00:.3f}')
                print(f'P(01): {p01:.3f}')
                print(f'P(10): {p10:.3f}')
                print(f'P(11): {p11:.3f}')
                print(f'P(0|0): {p0_0:.3f}')
                print(f'P(0|1): {p0_1:.3f}')
                print(f'P(1|0): {p1_0:.3f}')
                print(f'P(1|1): {p1_1:.3f}')

def main2():
    c_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.499, 0.501]
    x = ivs[0]

    for c in c_list:
        b_seq = ""
        for i in range (l):
            b1 = treshold_function(x, c)
            b_seq += str(b1)
            x = skew_bernouli_map(x, c)

        print(f"c:{c}", b_seq[:10])
        print("length", len(b_seq))
        # save
        os.makedirs('assignment2/{}'.format(c), exist_ok=True)
        with open(f'assignment2/{c}/2_{c}.txt', 'w') as f:
            f.write(b_seq)


if __name__ == "__main__": 
    print("No. 1\n\n")
    main()
    print("No. 2\n\n")
    main2()



# scp -r fathan@gw.dbms.tunnels:/nas.dbms/fathan/test/kelas/44135/assignment2 /Users/fathan/Downloads