import numpy as np
import matplotlib.pyplot as plt
import os

def threshold_function(x, t):  # threshold function for making 0 and 1 value
    return 0 if x < t else 1

def skew_bernouli_map(x, c): # Bernoulli mapping function
    if x < c:
        return (x / c)
    else:
        return (x - c) / (1 - c)
    
def plm3(x, p_1, p_2, t): # Markov plm3 mapping function
    def create_parameters(p_1, p_2):
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

        return a, a_positive, c1, c2, a1, a2
    
    a, a_positive, c1, c2, a1, a2 = create_parameters(p_1, p_2)

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

def memoryless_bernoulli():
    l = 1000000 # length (N)
    c = t = 0.49999 # t = c = 0.5 (~ 4.9999)
    def cte(p): # for sequence of errors with memoryless source (c=t=1-p)
        return 1 - p
    p_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49999]

    for p in p_list:
        x0 = 0.1782612
        z0 = 0.5673244
        derr = 0 # counter for error

        for i in range (l):
            # information source and coding : Bernoulli map (c = t)
            x1 = skew_bernouli_map(x0, c); x2  = skew_bernouli_map(x1, c)
            b0 = threshold_function(x0, t); b1 = threshold_function(x1, t); b2 = threshold_function(x2, t)
            b3 = b0 ^ b1 ^ b2
            x0 = skew_bernouli_map(x2, c) # prepare x0 for next loop

            # generating a sequence of errors with memoryless source (c=t=1-p) and information xor error
            z1 = skew_bernouli_map(z0, cte(p)); z2 = skew_bernouli_map(z1, cte(p)); z3 = skew_bernouli_map(z2, cte(p))
            e0 = threshold_function(z0, cte(p)); e1 = threshold_function(z1, cte(p)); e2 = threshold_function(z2, cte(p)); e3 = threshold_function(z3, cte(p)) # error sequence
            r0 = b0 ^ e0; r1 = b1 ^ e1; r2 = b2 ^ e2; r3 = b3 ^ e3; # received sequence
            if (e0 == 1)|(e1 == 1)|(e2 == 1)|(e3 == 1):
                er = 1
            else:
                er = 0
            z0 = skew_bernouli_map(z3, cte(p)) # prepare z0 for next loop

            # error-detecting and count the number of undetected errors
            derr = derr + (r0 ^ r1 ^ r2 ^ r3 ^ er)

        # probability of undetected errors
        computed_value = derr / l
        theoretical_value = 6 * p**2 * (1-p)**2 + p**4

        # print the result
        print(f'For p: {p}, computed value: {computed_value:.5f} and theoretical value: {theoretical_value:.5f}')

def markov():
    l = 1000000 # length (N)
    c = t = 0.49999 # t = c = 0.5 (~ 4.9999)
    def cte(p): # for sequence of errors with memoryless source (c=t=1-p)
        return 1 - p
    p_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49999]
    p2_list = [0.16, 0.34] # another parameter p2

    for p in p_list:
        for p2 in p2_list:
            p1 = p / (1 - p) * p2
            x0 = 0.1782612
            z0 = 0.5673244
            derr = 0 # counter for error

            for i in range (l):
                # information source and coding : Bernoulli map (c = t)
                x1 = skew_bernouli_map(x0, c); x2  = skew_bernouli_map(x1, c)
                b0 = threshold_function(x0, t); b1 = threshold_function(x1, t); b2 = threshold_function(x2, t)
                b3 = b0 ^ b1 ^ b2
                x0 = skew_bernouli_map(x2, c) # prepare x0 for next loop

                # generating a sequence of errors with markov-type source (c=t=1-p) and information xor error
                z1 = plm3(z0, p1, p2, cte(p)); z2 = plm3(z1, p1, p2, cte(p)); z3 = plm3(z2, p1, p2, cte(p))
                e0 = threshold_function(z0, cte(p)); e1 = threshold_function(z1, cte(p)); e2 = threshold_function(z2, cte(p)); e3 = threshold_function(z3, cte(p)) # error sequence
                r0 = b0 ^ e0; r1 = b1 ^ e1; r2 = b2 ^ e2; r3 = b3 ^ e3; # received sequence
                if (e0 == 1)|(e1 == 1)|(e2 == 1)|(e3 == 1):
                    er = 1
                else:
                    er = 0
                z0 = plm3(z3, p1, p2, cte(p)) # prepare z0 for next loop

                # error-detecting and count the number of undetected errors
                derr = derr + (r0 ^ r1 ^ r2 ^ r3 ^ er)

            # probability of undetected errors
            computed_value = derr / l
            theoretical_value = (p2 / (p1 + p2)) * ((1 - p1) * p1 * (1 - p2) + p1 * (1 - p2) * p2 + p1 * p2 * p1) + (p1 / (p1 + p2)) * (p2 * (1 - p1) * p1 + p2 * p1 * p2 + (1 - p2) * p2 * (1 - p1) + (1 - p2) * (1 - p2) * (1 - p2))

            # print the result
            print(f'For p: {p}, p1: {p1:.3f}, p2: {p2}; computed value: {computed_value:.5f} and theoretical value: {theoretical_value:.5f}')

if __name__ == '__main__':
    print('\nMemoryless with bernoulli map\n')
    memoryless_bernoulli()
    print('\nMarkov\n')
    markov()
