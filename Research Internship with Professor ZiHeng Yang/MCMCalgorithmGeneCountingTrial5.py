#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:10:09 2022

@author: bananabelyong
"""
import numpy as np
from numpy.random import random
import scipy
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# initialize parameters
nA = 44; nB = 27; nAB = 4; nO = 88
p = 0.3; q=0.3; r = 1 - p - q
n = nA + nB + nAB + nO
w = 0.1

x = [nA, nB, nO, nAB]
theta=[p, q, r]

"""
def priordistribution(x):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 0 for all invalid values of sigma 
    if(x[1] <=0):
        return 0
    return 1

#Calculates the likelihood of data with a given sigma (new/current) 
def manual_log_like_normal(x,observed_data):
    #x[0]=mu, x[1]=sigma (current)
    return np.sum(-np.log(x[1] * np.sqrt(2*np.pi) )-((observed_data-x[0])**2) / (2*x[1]**2))

def log_likelihood(x,observed_data):
    #x[0]=mu, x[1]=sigma (current)
    #data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0],x[1]).pdf(observed_data)))

#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # As log likelihood was implemented, exponentiate in order to compare with random number generated
        # Random number generation between 0 and 1, U(0,1)
        return (accept < (np.exp(x_new-x)))


    a) likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    b) transition_model(x): a function that draws a sample from a symmetric distribution
    b) (similar to sliding window proposal where t-w/2, t+w/2) and returns it
    c) param_init: a starting sample
    d) iterations: number of accepted to generated 
    e) data: data that is modelled 
    f) acceptance_rule(x,x_new): accept or reject the new sample 


def metropolis_hastings(likelihood_computer, prior, transition_model, parameters_1, iterations,data, acceptance_rule):
    x = parameters_1
    accepted = []
    rejected = []   
    for i in range(iterations):
        x_new =  transition_model(x)    
        x_likelihood = likelihood_computer(x,data)
        x_new_likelihood = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_likelihood + np.log(prior(x)),x_new_likelihood+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)
"""
def log_likelihood(theta, x):
    # unpack list
    nA = x[0]; nB = x[1]; nO = x[2]; nAB = x[3]
    p = theta[0]; q = theta[1]; r = theta[2]

    # calculate each part of the equation individually
    pt1 = nA * np.log(p**2 + 2*p*r)
    pt2 = nB * np.log(q**2 + 2*q*r)
    pt3 = nAB * np.log(2*p*q)
    pt4 = nO * np.log(r**2)

    return pt1 + pt2 + pt3 + pt4

def prior_likelihood(theta, x):
    #unpaack list
    nA = x[0]; nB = x[1]; nO = x[2]; nAB = x[3]
    p = theta[0]; q = theta[1]; r = theta[2]
    prob = ((p**2 + 2*p*r)**nA)*((q**2 + 2*q*r)**nB)*((2*p*q)**nAB)*((r**2)**nO)
    log_prob= np.log(prob)
    return prob

def posterior_distribution(prior, x):
    posterior= np.log(prior)*log_likelihood
    return posterior

def metropolis_hastings(theta, x, w, n):
    # 1. Initialize algorithm parameters (p, q, r, n..., h...
    #       Set acceptance counter to 0
    #       Create initial sample list
    #       Calculate intial logpriorlikelihood
    nA = x[0]; nB = x[1]; nO = x[2]; nAB = x[3]
    p = theta[0]; q = theta[1]; r = theta[2]
    accepted_values=[]
    accepted_values.append(theta)
    accept=0
    log_prior=prior_likelihood(theta, x)

    # 2. Start for loop for N iterations
    for i in range(10):
        
        # 3. Calculate new p, q, r and anything else
        #       Calculate new prior likelihood and use to find logratio
        pnew, qnew, rnew = genecounting(theta, n)
        log_new= prior_likelihood(theta=[pnew, qnew, rnew], x=x)
        logratio=log_new-log_prior
        
        # 4. if logratio >= 0 or random number is less than exp(logratio)
        #       Accept new p, q, r
        #       Accept new log likelihood
        #       Store new solution
        if logratio >=0 or random()<np.exp(logratio):
            p= pnew; q= qnew; r=rnew
            log_prior=log_new
            accept+=1
        accepted_values.append([p, q, r])    
    return accepted_values

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

    if e !=0:
        n=np.trunc(e/(b-a))
        if (n-2*np.trunc(n/2) !=0): #change sdide if n is odd
            side=1-side
        e = e -n*(b-a)
        if side==1:
            x=b-e
        else:
            x=a+e
    return(round(x, 3))

def genecounting(theta, n):
    p = theta[0]
    q = theta[1]
    r = theta[2]

    hA = p**2 / (p**2 + 2 * p * r)
    hB = q**2 / (q * 82 + 2 * q * r)


    p = 1 / (2 * n) * (nAB + nA * (1 + hA))
    q = 1 / (2 * n) * (nAB + nB * (1 + hB))

    r = 1 - p - q

    return p, q, r

def make_new_number(theta, w):
    # generate random number between 0 and 1
    u = random()

    p = theta[0]
    q = theta[1]
    r = theta[2]

    # if statements for ... 
    if u < 1./3.:
        s = p + q
        pnew = p + w * (random() - 1/2)
        pnew = reflect(pnew, 0, s)
        qnew = s - pnew
        rnew = r
    elif u < 2./3.:
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

print(f"Before: p:{p} q:{q} r:{r}")

plist = []; qlist = []; rlist = []; llist = []
for _ in range(10):
    #p, q, r = make_new_number(theta=[p, q, r], w=w)
    p, q, r  = genecounting(theta=[p, q, r], n=n)
    ll_value = log_likelihood(theta=[p, q, r], x=x) 

    plist.append(p)
    qlist.append(q)
    rlist.append(r)
    llist.append(ll_value)

print(f"After: p:{p} q:{q} r:{r}")

print('-'*25)
print("plist")
print(plist)
print('-'*25)
print("qlist")
print(qlist)
print('-'*25)
print("rlist")
print(rlist)
print('-'*25)
print("likelihood list")
print(llist)

# create plot
fig = plt.figure()
mpl.style.use("seaborn")
#plt.suptitle("Joint of p and q")
plt.plot(range(len(plist)), plist)
plt.xlabel('p')
plt.ylabel('q')
plt.show()
"""
mcmclist = metropolis_hastings(theta, x, w, n)
print(mcmclist)

Set hA adn Hb to 0.5, calculate the values of p and q 
once computed, go to equation 1.10. calculate ha and hb based on p and q. 
put this in a loop. once calculated, repeat the calculation, with new values of ha and hb
"""