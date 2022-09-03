#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 3 09:10:08 2022

@author: bananabelyong
"""
from imp import PY_COMPILED
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 3 09:10:08 2022

@author: bananabelyong
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn') #graphs will be more readable.

def equations(nA, nB, nAB, nO, hA, hB): #packaged phenotypes and hA, hB
    """
    Function is called equations to define equations 1.9 and 1.10 from ABO blood group paper to calculate p, q, r, and
    hA and hB before doing while loop.
    """
    # initialize sample size
    n = nA + nB + nAB + nO 

    # calculate p, q, r from given hA and hB from initilization. 
    p = (1/(2 * n)) * (nAB + nA * (1 + hA))
    q = (1/(2 * n)) * (nAB + nB * (1 + hB))
    r = 1 - p - q

    hA = p / (p + 2*r)
    hB = q / (q + 2*r)
    
    # returns parameters as two list
    return p, q, r, hA, hB

def log_likelihood(nA, nB, nAB, nO, p, q, r):
    """
    Code structure calculates the log likelihood according to ZiHeng's email comments 
    """
    # +=adds another value with the variable's value and assigns the new value to the variable
    # calculate log likelihood from A, B, AB and O count respectively. 
    lnL=nA * np.log(p**2 + 2*p*r) 
    lnL+=nB * np.log(q**2 + 2*q*r)
    lnL+=nAB * np.log(2*p*q)
    lnL+=nO * np.log(r**2)

    return lnL

def gene_counting(): #run gene counting algorithm in a loop
    # initialize parameters
    nA = 44; nB = 27; nAB = 4; nO = 88 #sample given from Morton (1964)
    hA = 0.5; hB = 0.5  # initial guesses for hA and hB

    abs_criteria = 1e-3   # loop termination criteria

    #initialize parameters and log likelihood
    p, q, r, hA, hB = equations(nA, nB, nAB, nO, hA, hB)
    p_old = p
    q_old = q
    r_old = r

    plist = []; qlist = []; rlist = []
    plist.append(p)
    qlist.append(q)
    rlist.append(r)

    lnL_list = []
    lnL_list.append(log_likelihood(nA, nB, nAB, nO, p, q, r))

    # run algorithm
    while True: #infinite loop 

        # estimate new parameter values 
        p, q, r, hA, hB = equations(nA, nB, nAB, nO, hA, hB)
        plist.append(p)
        qlist.append(q)
        rlist.append(r)

        # calculate log likelihood
        lnL_list.append(log_likelihood(nA, nB, nAB, nO, p, q, r))

        # find relative difference of old and new parameters p, q, r 
        pdiff = np.abs(p_old - p)
        qdiff = np.abs(q_old - q)
        rdiff = np.abs(r_old - r)

        # termination condition
        if (pdiff < abs_criteria and qdiff < abs_criteria) and rdiff < abs_criteria:
            break

        # if termination condition fails, store new parameter values as old values
        pold=p
        qold=q
        rold=r

    print("Log Likelihood Values")
    print(lnL_list)

    print('P values:')
    print(plist)
    print('Q values:')
    print(qlist)
    print('R values:')
    print(rlist)

    # plot p vs q 
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    
    ax.plot(plist, qlist, 'r-')
    ax.set_xlabel('p')
    ax.set_ylabel('q')

    plt.savefig(f'p_vs_q.png')

if __name__=="__main__":
    gene_counting()


