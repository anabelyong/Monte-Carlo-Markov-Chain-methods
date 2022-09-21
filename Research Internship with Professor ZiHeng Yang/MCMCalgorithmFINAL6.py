#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:10:09 2022

@author: bananabelyong
"""
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import sys

def prior(p, q): 
    #symmetric Dirichlet distribution
    if (p > 0 and q > 0) and p + q < 1:
        return 2 

def posterior_distribution(nA, nB, nAB, nO, p, q, r):
    #calculate posterior density
    posterior = np.log(prior(p, q)) * log_likelihood(nA, nB, nAB, nO, p, q, r)

    return posterior

def log_likelihood(nA, nB, nAB, nO, p, q, r):
    # calculate log likelihood from A, B, AB and O count respectively. 
    lnL=nA * np.log(p**2 + 2*p*r) 
    lnL+=nB * np.log(q**2 + 2*q*r)
    lnL+=nAB * np.log(2*p*q)
    lnL+=nO * np.log(r**2)

    return lnL

def reflect(x, a, b):
    # returns the values when x is reflected into the range (a,b)
    side = 0 
    e = 0

    if x < a:
        e = a - x
        side = 0
    elif x > b:
        e = x - b
        side = 1 

    if e!=0:
        n=np.trunc(e/(b-a))
        if (n-2*np.trunc(n/2) !=0): #change sdide if n is odd
            side=1-side
        e = e -n*(b-a)
        if side==1:
            x=b-e
        else:
            x=a+e
    return(round(x, 3))

def random_generator(p, q, r):
    # generate random number between 0 and 1
    u = random()

    if u < 1/3:
        s = p + q
        pnew = p + w * (random() - 1/2)
        pnew = reflect(pnew, 0, s)
        qnew = s - pnew
        rnew = r
    elif u < 2/3:
        s = q + r
        qnew = q + w * (random() - 1/2)
        qnew = reflect(qnew, 0, s)
        rnew = s - qnew
        pnew = p
    else:    
        s = r + p
        rnew = r + w * (random() - 1/2)
        rnew = reflect(rnew, 0, s)
        pnew = s - rnew
        qnew = q
    
    return pnew, qnew, rnew

def mcmc_algorithm(nA, nB, nAB, nO, p, q, r, sample_size):
    # intialize 
    lnP = log_likelihood(nA, nB, nAB, nO, p, q, r)

    # list to store accepted and unaccepted samples (merged list)
    samples=[]

    #start algorithm
    acceptance_counter = 0  # count number of times new paramaters have been accepted
    for _ in range(sample_size):
        pnew, qnew, rnew = random_generator(p, q, r)

        #calculation of acceptance ratio 
        lnPnew  = log_likelihood(nA, nB, nAB, nO, pnew, qnew, rnew)
        lnalpha = lnPnew - lnP
        if lnalpha > 0 or np.exp(lnalpha) > random():
            p = pnew
            q = qnew
            r = rnew
            lnP = lnPnew
            acceptance_counter += 1
        samples.append([pnew, qnew, rnew])

    return samples, acceptance_counter
    
if __name__=="__main__":
    # initialize parameters
    nA = 44; nB = 27; nAB = 4; nO = 88
    p = 0.3; q = 0.3; r = 1 - p - q
    n = nA + nB + nAB + nO
    w = 0.1
    
    if len(sys.argv) != 3:
        print(" usage: " + sys.argv[0] + " [BURNIN] [SAMPLES]")
        sys.exit(0)

    # burnin samples
    initial_samples, _ = mcmc_algorithm(nA, nB, nAB, nO, p, q, r, int(sys.argv[1]))

    p = initial_samples[-1][0]
    q = initial_samples[-1][1]
    r = initial_samples[-1][2]

    # run mcmc algorithms
    samples_list, accept_counter = mcmc_algorithm(nA, nB, nAB, nO, p, q, r, int(sys.argv[2]))
    
    # unpack
    plist = []; qlist = []; rlist=[]
    for i in range(len(samples_list)):
        plist.append(samples_list[i][0])
        qlist.append(samples_list[i][1])
        rlist.append(samples_list[i][2])

    # create posterior calculation list
    post = [posterior_distribution(nA, nB, nAB, nO, p, q, r) for p, q, r in zip(plist, qlist, rlist)]

    # calculate acceptance probability
    acceptance_proportion=accept_counter/(int(sys.argv[2]))
    print("Acceptance proportion P: ", acceptance_proportion)

    acceptance_proportion=accept_counter/(int(sys.argv[2]))
    print("Acceptance proportion Q: ", acceptance_proportion)

    acceptance_proportion=accept_counter/(int(sys.argv[2]))
    print("Acceptance proportion R: ", acceptance_proportion)

    fig1 = plt.figure()
    plt.hist(plist)
    plt.xlabel('p')
    plt.ylabel('Unnormalized posterior density')

    fig2 = plt.figure()
    plt.scatter(plist, qlist, s=1) 
    plt.xlabel('p')
    plt.ylabel('q')

    plt.show()




