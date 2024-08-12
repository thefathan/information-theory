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
        x0 = 0.1782612 # initial value for sequence
        z0 = 0.5673244 # initial value for error sequence
        ok = 0 # counter for correct decoding
        berr0 = 0 # counter for error probability (before decoding)
        blerr = 0 # counter for incorrect decoding
        berr = 0 # counter for error probability (after decoding)

        for i in range (l):
            # information source and coding : Bernoulli map (c = t)
            x1 = skew_bernouli_map(x0, c); x2  = skew_bernouli_map(x1, c); x3 = skew_bernouli_map(x2, c)
            b0 = threshold_function(x0, t); b1 = threshold_function(x1, t); b2 = threshold_function(x2, t); b3 = threshold_function(x3, c)
            b4 = b0 ^ b1 ^ b2; b5 = b0 ^ b1 ^ b3; b6 = b0 ^ b2 ^ b3
            x0 = skew_bernouli_map(x3, c) # prepare x0 for next loop

            # generating a sequence of errors with memoryless source (c=t=1-p) and information xor error
            z1 = skew_bernouli_map(z0, cte(p)); z2 = skew_bernouli_map(z1, cte(p)); z3 = skew_bernouli_map(z2, cte(p)); z4 = skew_bernouli_map(z3, cte(p)); z5 = skew_bernouli_map(z4, cte(p)); z6 = skew_bernouli_map(z5, cte(p))
            e0 = threshold_function(z0, cte(p)); e1 = threshold_function(z1, cte(p)); e2 = threshold_function(z2, cte(p)); e3 = threshold_function(z3, cte(p)); e4 = threshold_function(z4, cte(p)); e5 = threshold_function(z5, cte(p)); e6 = threshold_function(z6, cte(p)) # error sequence
            r0 = b0 ^ e0; r1 = b1 ^ e1; r2 = b2 ^ e2; r3 = b3 ^ e3; r4 = b4 ^ e4; r5 = b5 ^ e5; r6 = b6 ^ e6 # received sequence
            z0 = skew_bernouli_map(z6, cte(p)) # prepare z0 for next loop

            # ecount the number of error bits (before decoding)
            berr0 = berr0 + e0 + e1 + e2 + e3 + e4 + e5 + e6

            # calculation of the syndrome
            s0 = r0 ^ r1 ^ r2 ^ r4; s1 = r0 ^ r1 ^ r3 ^ r5; s2 = r0 ^ r2 ^ r3 ^ r6

            # error-correcting based on the syndrome
            if ((s0 == 1) & (s1 == 1) & (s2 == 1)):
                r0 = r0 ^ 1
            elif ((s0 == 1) & (s1 == 1) & (s2 == 0)):
                r1 = r1 ^ 1
            elif ((s0 == 1) & (s1 == 0) & (s2 == 1)):
                r2 = r2 ^ 1
            elif ((s0 == 0) & (s1 == 1) & (s2 == 1)):
                r3 = r3 ^ 1
            elif ((s0 == 1) & (s1 == 0) & (s2 == 0)):
                r4 = r4 ^ 1
            elif ((s0 == 0) & (s1 == 1) & (s2 == 0)):
                r5 = r5 ^ 1
            elif ((s0 == 0) & (s1 == 0) & (s2 == 1)):
                r6 = r6 ^ 1

            # count the number of incorrect decoding
            if ((r0 == b0) & (r1 == b1) & (r2 == b2) & (r3 == b3) & (r4 == b4) & (r5 == b5) & (r6 == b6)):
                ok += 1
            else:
                blerr += 1

            # count the number of error bits (after decoding)
            berr = berr + (r0 ^ b0) + (r1 ^ b1) + (r2 ^ b2) + (r3 ^ b3) + (r4 ^ b4) + (r5 ^ b5) + (r6 ^ b6)

        # probability of incorect decoding
        incorrect_computed_value = blerr / l
        correct_theoretical_value = 7 * p * (1 - p)**6 + (1 - p)**7
        incorrect_theoretical_value = 1 - correct_theoretical_value

        # probability of bit error (before and after decoding)
        bit_error_before = berr0 / (7 * l)
        bit_error_after = berr / (7 * l)

        # print the result
        print(f'For p: {p}\nINCORRECT DECODING; computed value: {incorrect_computed_value:.5f} and theoretical value: {incorrect_theoretical_value:.5f}\nPROBABILITY BIT ERROR; before: {bit_error_before:.5f} and after: {bit_error_after:.5f}\n\n')

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
            x0 = 0.1782612 # initial value for sequence
            z0 = 0.5673244 # initial value for error sequence
            ok = 0 # counter for correct decoding
            berr0 = 0 # counter for error probability (before decoding)
            blerr = 0 # counter for incorrect decoding
            berr = 0 # counter for error probability (after decoding)

            for i in range (l):
                # information source and coding : Bernoulli map (c = t)
                x1 = skew_bernouli_map(x0, c); x2  = skew_bernouli_map(x1, c); x3 = skew_bernouli_map(x2, c)
                b0 = threshold_function(x0, t); b1 = threshold_function(x1, t); b2 = threshold_function(x2, t); b3 = threshold_function(x3, c)
                b4 = b0 ^ b1 ^ b2; b5 = b0 ^ b1 ^ b3; b6 = b0 ^ b2 ^ b3
                x0 = skew_bernouli_map(x3, c) # prepare x0 for next loop

                # generating a sequence of errors with markov-type source (c=t=1-p) and information xor error
                z1 = plm3(z0, p1, p2, cte(p)); z2 = plm3(z1, p1, p2, cte(p)); z3 = plm3(z2, p1, p2, cte(p)); z4 = plm3(z3, p1, p2, cte(p)); z5 = plm3(z4, p1, p2, cte(p)); z6 = plm3(z5, p1, p2, cte(p))
                e0 = threshold_function(z0, cte(p)); e1 = threshold_function(z1, cte(p)); e2 = threshold_function(z2, cte(p)); e3 = threshold_function(z3, cte(p)); e4 = threshold_function(z4, cte(p)); e5 = threshold_function(z5, cte(p)); e6 = threshold_function(z6, cte(p)) # error sequence
                r0 = b0 ^ e0; r1 = b1 ^ e1; r2 = b2 ^ e2; r3 = b3 ^ e3; r4 = b4 ^ e4; r5 = b5 ^ e5; r6 = b6 ^ e6 # received sequence
                z0 = plm3(z6, p1, p2, cte(p)) # prepare z0 for next loop

                # ecount the number of error bits (before decoding)
                berr0 = berr0 + e0 + e1 + e2 + e3 + e4 + e5 + e6

                # calculation of the syndrome
                s0 = r0 ^ r1 ^ r2 ^ r4; s1 = r0 ^ r1 ^ r3 ^ r5; s2 = r0 ^ r2 ^ r3 ^ r6

                # error-correcting based on the syndrome
                if ((s0 == 1) & (s1 == 1) & (s2 == 1)):
                    r0 = r0 ^ 1
                elif ((s0 == 1) & (s1 == 1) & (s2 == 0)):
                    r1 = r1 ^ 1
                elif ((s0 == 1) & (s1 == 0) & (s2 == 1)):
                    r2 = r2 ^ 1
                elif ((s0 == 0) & (s1 == 1) & (s2 == 1)):
                    r3 = r3 ^ 1
                elif ((s0 == 1) & (s1 == 0) & (s2 == 0)):
                    r4 = r4 ^ 1
                elif ((s0 == 0) & (s1 == 1) & (s2 == 0)):
                    r5 = r5 ^ 1
                elif ((s0 == 0) & (s1 == 0) & (s2 == 1)):
                    r6 = r6 ^ 1
                else:
                    None

                # count the number of incorrect decoding
                if ((r0 == b0) & (r1 == b1) & (r2 == b2) & (r3 == b3) & (r4 == b4) & (r5 == b5) & (r6 == b6)):
                    ok += 1
                else:
                    blerr += 1

                # count the number of error bits (after decoding)
                berr = berr + (r0 ^ b0) + (r1 ^ b1) + (r2 ^ b2) + (r3 ^ b3) + (r4 ^ b4) + (r5 ^ b5) + (r6 ^ b6)

            # probability of incorect decoding
            incorrect_computed_value = blerr / l
            correct_theoretical_value = p1 / (p1 + p2) * p2 * (1 - p1)**5 + 5 * p2 / (p1 + p2) * p1 * p2 * (1 - p1)**4 + p2 / (p1 + p2) * (1 - p1)**5 * p1 + p2 / (p1 + p2) * (1 - p1)**6
            incorrect_theoretical_value = 1 - correct_theoretical_value

            # probability of bit error (before and after decoding)
            bit_error_before = berr0 / (7 * l)
            bit_error_after = berr / (7 * l)

            # print the result
            print(f'For p: {p}, p1: {p1:.3f}, p2: {p2}\nINCORRECT DECODING; computed value: {incorrect_computed_value:.5f} and theoretical value: {incorrect_theoretical_value:.5f}\nPROBABILITY BIT ERROR; before: {bit_error_before:.5f} and after: {bit_error_after:.5f}\n\n')

if __name__ == '__main__':
    print('\nNo. 1. Memoryless with bernoulli map\n')
    memoryless_bernoulli()
    print('\nNo. 2. Markov\n')
    markov()
