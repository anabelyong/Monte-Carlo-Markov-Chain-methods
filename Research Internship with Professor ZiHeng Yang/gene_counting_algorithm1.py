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

def estimation_equations(nlist, hlist): #packaged phenotypes and hA, hB
    """
    Function is called equations to define equations 1.9 and 1.10 from ABO blood group paper to calculate p, q, r, and
    hA and hB before doing while loop.
    """
    # unpack input arguments
    nA = nlist[0]; nB = nlist[1]; nAB = nlist[2]; nO = nlist[3]
    hA = hlist[0]; hB = hlist[1]

    # initialize sample size
    n = nA + nB + nAB + nO 

    # calculate p, q, r from given hA and hB from initilization. 
    p = (1/(2 * n)) * (nAB + nA * (1 + hA))
    q = (1/(2 * n)) * (nAB + nB * (1 + hB))
    r = 1 - p - q

    hA = p / (p + 2*r)
    hB = q / (q + 2*r)
    
    # returns parameters as two list
    return [p, q, r], [hA, hB]

def log_likelihood(theta, nlist):
    """
    Code structure calculates the log likelihood according to ZiHeng's email comments 
    """
    #unpack list
    nA = nlist[0]; nB = nlist[1]; nAB = nlist[2]; nO = nlist[3]
    p = theta[0]; q = theta[1]; r = theta[2]

    # +=adds another value with the variable's value and assigns the new value to the variable
    # calculate log likelihood from A, B, AB and O count respectively. 
    lnL  = nA * np.log(p**2 + 2*p*r) 
    lnL +=nB * np.log(q**2 + 2*q*r)
    lnL += nAB * np.log(2*p*q)
    lnL += nO * np.log(r**2)

    return lnL

def gene_counting(): #run gene counting algorithm in a loop
    # initialize parameters
    nA = 44; nB = 27; nAB = 4; nO = 88 #sample given from Morton (1964)
    hA = 0.5; hB = 0.5  # initial guesses for hA and hB

    rel_criteria = 1e-4   # loop termination criteria

    #pack parameters into list
    nlist = [nA, nB, nAB, nO]
    hlist = [hA, hB]

    #initialize parameters and log likelihood
    theta, hlist = estimation_equations(nlist, hlist)
    theta_old    = theta    # store parameters in a old list for termination criteria

    theta_list = [] # to store theta values
    theta_list.append(theta)

    lnL_list = []
    lnL_list.append(log_likelihood(theta, nlist))

    # run algorithm
    while True: #infinite loop 

        # estimate new parameter values 
        theta, hlist = estimation_equations(nlist, hlist)
        theta_list.append(theta)

        # calculate log likelihood
        lnL_list.append(log_likelihood(theta, nlist))

        # find relative difference of old and new parameters p, q, r 
        rel_diff = []   # list to store relative difference between old and new parameters
        for i in range(len(theta)):
            rel_diff.append(np.abs((theta[i] - theta_old[i])/max(theta_old[i], theta[i])))

        # termination condition
        if all(rel_value < rel_criteria for rel_value in rel_diff):
            break

        # if termination condition fails, store new parameter values as old values
        theta_old = theta

    print("Log Likelihood Values")
    print(lnL_list)

    # separate p, q, r
    p_list = []; q_list = []; r_list = []
    for i in range(len(theta_list)):
        p_list.append(theta_list[i][0])
        q_list.append(theta_list[i][1])
        r_list.append(theta_list[i][2])
    
    print('P values:')
    print(p_list)
    print('Q values:')
    print(q_list)
    print('R values:')
    print(r_list)

    # plot p vs q 
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    
    ax.plot(p_list, q_list, 'r-')
    ax.set_xlabel('p')
    ax.set_ylabel('q')

    plt.show()

if __name__=="__main__":
    gene_counting()
