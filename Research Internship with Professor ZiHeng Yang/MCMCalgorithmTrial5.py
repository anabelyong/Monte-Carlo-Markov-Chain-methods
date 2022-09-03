#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:10:09 2022

@author: bananabelyong

"""

import numpy as np
import scipy
import scipy.stats
import matplotlib as mpl   
import matplotlib.pyplot as plt

mod1=lambda t:np.random.normal(t)

#form a sample of 163 samples (n), initialize an array for parameters theta and x
population = mod1(163)

"""
x= [nA, nB, nO, nAB]
theta=[pA, pB, pO]
"""

#transition model to move from sigma_current to sigma_new
transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))[0]]

def priordistribution(x):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 0 for all invalid values of sigma 
    if(x[1] <=0):
        return 0
    return 1

#Calculates the likelihood of data with a given sigma (new/current) 
def manual_log_like_normal(x,observed_data):
    #x[0]=mu, x[1]=sigma (current)
    return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((observed_data-x[0])**2) / (2*x[1]**2))

"""
def log_likelihood(x,observed_data):
    #x[0]=mu, x[1]=sigma (current)
    #data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0],x[1]).pdf(observed_data)))
"""

#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # As log likelihood was implemented, exponentiate in order to compare with random number generated
        # Random number generation between 0 and 1, U(0,1)
        return (accept < (np.exp(x_new-x)))

"""
    a) likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    b) transition_model(x): a function that draws a sample from a symmetric distribution
    b) (similar to sliding window proposal where t-w/2, t+w/2) and returns it
    c) param_init: a starting sample
    d) iterations: number of accepted to generated 
    e) data: data that is modelled 
    f) acceptance_rule(x,x_new): accept or reject the new sample 
"""
def metropolis_hastings(likelihood_computer,prior, transition_model, parameters_1,iterations,data,acceptance_rule):
    
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
Plot 
show=int(-0.75*accepted.shape[0])
hist_show=int(-0.75*accepted.shape[0])

fig = plt.figure(figsize=(20,10))

\
"""