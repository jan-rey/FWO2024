#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of temperature with policy-gradient method
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    adversarial: least frequent 60% of sequences
                         
Critical parameters: temperature, learning rate, unchosen addition (there is no chosen here)


@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random

#simulation of Rescorla-Wagner model with meta-learning of epsilon 
#meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#rewards are baselined
def simulate_RW_MLinv_temp_adversarial(x, Q_alpha, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of inv_temp
    #ML_std_int         --->        initial value for the standard deviation of inv_temp
    #scale              --->        scale with which ML-parameter gets transformed to inv_temp in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on temp
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    seq_options2 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq2 = np.random.uniform(0.9,1.1,K_seq)

    seq_options3 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq3 = np.random.uniform(0.9,1.1,K_seq)

    seq_options4 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq4 = np.random.uniform(0.9,1.1,K_seq)
    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    inv_temp_var = np.zeros((T), dtype=float)
    inv_temp_var_mean = np.zeros((T), dtype=float)
    inv_temp_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(inv_temp_mean_int/(1-inv_temp_mean_int))
    #ML_std_int = (1/scale)*np.log(inv_temp_std_int/(1-inv_temp_std_int))

    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(250)
    for i in range(250):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<2400]    

    for t in range(T):
        #temperature is calculated with a logit transformation of the ML
        inv_temp = ML #np.exp(scale*ML)/(1+np.exp(scale*ML))
        inv_temp_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        inv_temp_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        inv_temp_var[t] = inv_temp
        inv_temp_var_mean[t] = inv_temp_mean
        inv_temp_var_std[t] = inv_temp_std
        # store values for Q and inverse temperature
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options
        sum_soft = 0
        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
        
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)

        #variable
    
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0
            '''
            sorted_freq = Freq1.copy()
            sorted_freq.sort()
            sorted_index = np.where(sorted_freq== current_freq)[0]
            if sorted_index < 38:
                r[t] = 1
            else:
                r[t] = 0
            '''
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984


        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,100])
            ML_mean = np.max([ML_mean, 0])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, inv_temp_var, inv_temp_var_mean, inv_temp_var_std, ML_stored, ML_mean_stored, ML_std_stored

#simulation of Rescorla-Wagner model with meta-learning of epsilon 
#meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#rewards are baselined
def simulate_RW_MLinv_temp_stable(x, Q_alpha, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of inv_temp
    #ML_std_int         --->        initial value for the standard deviation of inv_temp
    #scale              --->        scale with which ML-parameter gets transformed to inverse temp in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on temp
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    seq_options2 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq2 = np.random.uniform(0.9,1.1,K_seq)

    seq_options3 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq3 = np.random.uniform(0.9,1.1,K_seq)

    seq_options4 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq4 = np.random.uniform(0.9,1.1,K_seq)
    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    inv_temp_var = np.zeros((T), dtype=float)
    inv_temp_var_mean = np.zeros((T), dtype=float)
    inv_temp_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(inv_temp_mean_int/(1-inv_temp_mean_int))
    #ML_std_int = (1/scale)*np.log(inv_temp_std_int/(1-inv_temp_std_int))

    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(250)
    for i in range(250):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<2400]    

    for t in range(T):
        #inverse temp is calculated with a logit transformation of the ML
        inv_temp = ML #np.exp(scale*ML)/(1+np.exp(scale*ML))
        inv_temp_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        inv_temp_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        inv_temp_var[t] = inv_temp
        inv_temp_var_mean[t] = inv_temp_mean
        inv_temp_var_std[t] = inv_temp_std
        # store values for Q and inverse temp
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options
        sum_soft = 0
        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
        
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)

        a1 = reward_stable[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])
        
        
        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,100])
            ML_mean = np.max([ML_mean, 0])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, inv_temp_var, inv_temp_var_mean, inv_temp_var_std, ML_stored, ML_mean_stored, ML_std_stored



#simulation of Rescorla-Wagner model with meta-learning of epsilon 
#meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#rewards are baselined
def simulate_RW_MLinv_temp_volatile(x, Q_alpha, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of inv_temp
    #ML_std_int         --->        initial value for the standard deviation of inv_temp
    #scale              --->        scale with which ML-parameter gets transformed to inv temp in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on inv temp
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    seq_options2 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq2 = np.random.uniform(0.9,1.1,K_seq)

    seq_options3 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq3 = np.random.uniform(0.9,1.1,K_seq)

    seq_options4 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq4 = np.random.uniform(0.9,1.1,K_seq)
    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    inv_temp_var = np.zeros((T), dtype=float)
    inv_temp_var_mean = np.zeros((T), dtype=float)
    inv_temp_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(inv_temp_mean_int/(1-inv_temp_mean_int))
    #ML_std_int = (1/scale)*np.log(inv_temp_std_int/(1-inv_temp_std_int))

    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000]    

    for t in range(T):
        #inv temp is calculated with a logit transformation of the ML
        inv_temp = ML #np.exp(scale*ML)/(1+np.exp(scale*ML))
        inv_temp_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        inv_temp_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        inv_temp_var[t] = inv_temp
        inv_temp_var_mean[t] = inv_temp_mean
        inv_temp_var_std[t] = inv_temp_std
        # store values for Q and inv temp
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options
        sum_soft = 0
        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
        
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)
        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])
        
        
        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,100])
            ML_mean = np.max([ML_mean, 0])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, inv_temp_var, inv_temp_var_mean, inv_temp_var_std, ML_stored, ML_mean_stored, ML_std_stored


sim_nr = 'fwo2'
amount_of_sim = 100
x = 10000
T = 1*x
K = 8

reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_st = '3-5 70-30'
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
reward_vl = '3-5 90-10'

Q_alpha = 0.5
inv_temp = 10
unchosen = 0

inv_temp_mean_int = 10
MLinv_temp_mean_int = inv_temp_mean_int #np.log(inv_temp_mean_int/(1-inv_temp_mean_int))
lr_mean_int = 0.5
MLlr_mean_int = np.log(lr_mean_int/(1-lr_mean_int))
uc_mean_int = 0
MLuc_mean_int = uc_mean_int
ML_std_int = 1 #np.log(std_int/(1-std_int))

update = 10
Q_int = 1
reward_alpha = 0.25
ML_alpha_mean = 0.5
ML_alpha_std = 0.1 #LR std, LR pos and LR neg

#################################################################
#SIMULATIONS
#################################################################
'''
# META-LEARNING OF INVTEMPERATURE IN A VARIABLE CONTEXT (VARIABILITY IS REINFORCED)
#for barplots:
total_inv_temp_var = np.zeros(amount_of_sim)
total_inv_temp_var_mean = np.zeros(amount_of_sim)
total_inv_temp_var_std = np.zeros(amount_of_sim)
reward_inv_temp_var = np.zeros(amount_of_sim)

#for time plots:
r_inv_temp_var_cumsum = np.zeros(T)
r_inv_temp_var = np.zeros(T)
invtemperature_var = np.zeros(T)
invtemperature_mean_var = np.zeros(T)
invtemperature_std_var = np.zeros(T)

#for time plots of untransformed ML
MLinv_temp_var = np.zeros(T)
MLinv_temp_mean_var = np.zeros(T)
MLinv_temp_std_var = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_var, inv_temp_var_mean, inv_temp_var_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_adversarial(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_inv_temp_var[sim] = np.mean(inv_temp_var)
    total_inv_temp_var_mean[sim] = np.mean(inv_temp_var_mean)
    total_inv_temp_var_std[sim] = np.mean(inv_temp_var_std)
    reward_inv_temp_var[sim] = np.mean(r)

    #for time plot:
    r_inv_temp_cumsum = np.cumsum(r)
    r_var_inv_temp_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_var_inv_temp_cumsum_av[nr] = r_inv_temp_cumsum[nr]/divide

    r_inv_temp_var_cumsum = r_inv_temp_var_cumsum + r_var_inv_temp_cumsum_av
    r_inv_temp_var = r_inv_temp_var + r
    invtemperature_var = invtemperature_var + inv_temp_var
    invtemperature_mean_var = invtemperature_mean_var + inv_temp_var_mean
    invtemperature_std_var = invtemperature_std_var + inv_temp_var_std

    #for time plots of untransformed ML
    MLinv_temp_var = MLinv_temp_var + ML_stored
    MLinv_temp_mean_var = MLinv_temp_mean_var + ML_mean_stored
    MLinv_temp_std_var = MLinv_temp_std_var + ML_std_stored

#for bar plot:
av_inv_temp_var = np.mean(total_inv_temp_var)
av_inv_temp_var_mean = np.mean(total_inv_temp_var_mean)
av_inv_temp_var_std = np.mean(total_inv_temp_var_std)
av_inv_temp_reward_var = np.mean(reward_inv_temp_var)

std_inv_temp_var = np.std(total_inv_temp_var)
std_inv_temp_var_mean = np.std(total_inv_temp_var_mean)
std_inv_temp_var_std = np.std(total_inv_temp_var_std)
std_inv_temp_reward_var = np.std(reward_inv_temp_var)

ste_inv_temp_var = std_inv_temp_var/np.sqrt(amount_of_sim)
ste_inv_temp_var_mean = std_inv_temp_var_mean/np.sqrt(amount_of_sim)
ste_inv_temp_var_std = std_inv_temp_var_std/np.sqrt(amount_of_sim)
ste_inv_temp_reward_var = std_inv_temp_reward_var/np.sqrt(amount_of_sim)

#for time plot:
r_var_inv_temp_cumsum_end = np.divide(r_inv_temp_var_cumsum, amount_of_sim)
r_var_inv_temp_end = np.divide(r_inv_temp_var, amount_of_sim)
invtemperature_var_end= np.divide(invtemperature_var, amount_of_sim)
invtemperature_mean_var_end = np.divide(invtemperature_mean_var, amount_of_sim)
invtemperature_std_var_end = np.divide(invtemperature_std_var, amount_of_sim)

#for time plots of untransformed ML
MLinv_temp_var_end = np.divide(MLinv_temp_var, amount_of_sim)
MLinv_temp_mean_var_end = np.divide(MLinv_temp_mean_var, amount_of_sim)
MLinv_temp_std_var_end = np.divide(MLinv_temp_std_var, amount_of_sim)










# META-LEARNING OF INVTEMPERATURE IN A STABLE CONTEXT (VARIABILITY IS REINFORCED)
total_inv_temp_sta = np.zeros(amount_of_sim)
total_inv_temp_sta_mean = np.zeros(amount_of_sim)
total_inv_temp_sta_std = np.zeros(amount_of_sim)
reward_inv_temp_sta = np.zeros(amount_of_sim)

#for time plots:
r_inv_temp_sta_cumsum = np.zeros(T)
r_inv_temp_sta = np.zeros(T)
invtemperature_sta = np.zeros(T)
invtemperature_mean_sta = np.zeros(T)
invtemperature_std_sta = np.zeros(T)

#for time plots of untransformed ML
MLinv_temp_sta = np.zeros(T)
MLinv_temp_mean_sta = np.zeros(T)
MLinv_temp_std_sta = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_sta, inv_temp_sta_mean, inv_temp_sta_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_stable(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_inv_temp_sta[sim] = np.mean(inv_temp_sta)
    total_inv_temp_sta_mean[sim] = np.mean(inv_temp_sta_mean)
    total_inv_temp_sta_std[sim] = np.mean(inv_temp_sta_std)
    reward_inv_temp_sta[sim] = np.mean(r)

    #for time plot:
    r_inv_temp_cumsum = np.cumsum(r)
    r_sta_inv_temp_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_sta_inv_temp_cumsum_av[nr] = r_inv_temp_cumsum[nr]/divide

    r_inv_temp_sta_cumsum = r_inv_temp_sta_cumsum + r_sta_inv_temp_cumsum_av
    r_inv_temp_sta = r_inv_temp_sta + r
    invtemperature_sta = invtemperature_sta + inv_temp_sta
    invtemperature_mean_sta = invtemperature_mean_sta + inv_temp_sta_mean
    invtemperature_std_sta = invtemperature_std_sta + inv_temp_sta_std

    #for time plots of untransformed ML
    MLinv_temp_sta = MLinv_temp_sta + ML_stored
    MLinv_temp_mean_sta = MLinv_temp_mean_sta + ML_mean_stored
    MLinv_temp_std_sta = MLinv_temp_std_sta + ML_std_stored

#for bar plot:
av_inv_temp_sta = np.mean(total_inv_temp_sta)
av_inv_temp_sta_mean = np.mean(total_inv_temp_sta_mean)
av_inv_temp_sta_std = np.mean(total_inv_temp_sta_std)
av_inv_temp_reward_sta = np.mean(reward_inv_temp_sta)

std_inv_temp_sta = np.std(total_inv_temp_sta)
std_inv_temp_sta_mean = np.std(total_inv_temp_sta_mean)
std_inv_temp_sta_std = np.std(total_inv_temp_sta_std)
std_inv_temp_reward_sta = np.std(reward_inv_temp_sta)

ste_inv_temp_sta = std_inv_temp_sta/np.sqrt(amount_of_sim)
ste_inv_temp_sta_mean = std_inv_temp_sta_mean/np.sqrt(amount_of_sim)
ste_inv_temp_sta_std = std_inv_temp_sta_std/np.sqrt(amount_of_sim)
ste_inv_temp_reward_sta = std_inv_temp_reward_sta/np.sqrt(amount_of_sim)

#for time plot:
r_sta_inv_temp_cumsum_end = np.divide(r_inv_temp_sta_cumsum, amount_of_sim)
r_sta_inv_temp_end = np.divide(r_inv_temp_sta, amount_of_sim)
invtemperature_sta_end= np.divide(invtemperature_sta, amount_of_sim)
invtemperature_mean_sta_end = np.divide(invtemperature_mean_sta, amount_of_sim)
invtemperature_std_sta_end = np.divide(invtemperature_std_sta, amount_of_sim)

#for time plots of untransformed ML
MLinv_temp_sta_end = np.divide(MLinv_temp_sta, amount_of_sim)
MLinv_temp_mean_sta_end = np.divide(MLinv_temp_mean_sta, amount_of_sim)
MLinv_temp_std_sta_end = np.divide(MLinv_temp_std_sta, amount_of_sim)






# META-LEARNING OF INVTEMPERATURE IN A VOLATILE CONTEXT (VARIABILITY IS REINFORCED)
total_inv_temp_vol = np.zeros(amount_of_sim)
total_inv_temp_vol_mean = np.zeros(amount_of_sim)
total_inv_temp_vol_std = np.zeros(amount_of_sim)
reward_inv_temp_vol = np.zeros(amount_of_sim)

#for time plots:
r_inv_temp_vol_cumsum = np.zeros(T)
r_inv_temp_vol = np.zeros(T)
invtemperature_vol = np.zeros(T)
invtemperature_mean_vol = np.zeros(T)
invtemperature_std_vol = np.zeros(T)

#for time plots of untransformed ML
MLinv_temp_vol = np.zeros(T)
MLinv_temp_mean_vol = np.zeros(T)
MLinv_temp_std_vol = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_vol, inv_temp_vol_mean, inv_temp_vol_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_volatile(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_inv_temp_vol[sim] = np.mean(inv_temp_vol)
    total_inv_temp_vol_mean[sim] = np.mean(inv_temp_vol_mean)
    total_inv_temp_vol_std[sim] = np.mean(inv_temp_vol_std)
    reward_inv_temp_vol[sim] = np.mean(r)

    #for time plot:
    r_inv_temp_cumsum = np.cumsum(r)
    r_vol_inv_temp_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_vol_inv_temp_cumsum_av[nr] = r_inv_temp_cumsum[nr]/divide

    r_inv_temp_vol_cumsum = r_inv_temp_vol_cumsum + r_vol_inv_temp_cumsum_av
    r_inv_temp_vol = r_inv_temp_vol + r
    invtemperature_vol = invtemperature_vol + inv_temp_vol
    invtemperature_mean_vol = invtemperature_mean_vol + inv_temp_vol_mean
    invtemperature_std_vol = invtemperature_std_vol + inv_temp_vol_std

    #for time plots of untransformed ML
    MLinv_temp_vol = MLinv_temp_vol + ML_stored
    MLinv_temp_mean_vol = MLinv_temp_mean_vol + ML_mean_stored
    MLinv_temp_std_vol = MLinv_temp_std_vol + ML_std_stored

#for bar plot:
av_inv_temp_vol = np.mean(total_inv_temp_vol)
av_inv_temp_vol_mean = np.mean(total_inv_temp_vol_mean)
av_inv_temp_vol_std = np.mean(total_inv_temp_vol_std)
av_inv_temp_reward_vol = np.mean(reward_inv_temp_vol)

std_inv_temp_vol = np.std(total_inv_temp_vol)
std_inv_temp_vol_mean = np.std(total_inv_temp_vol_mean)
std_inv_temp_vol_std = np.std(total_inv_temp_vol_std)
std_inv_temp_reward_vol = np.std(reward_inv_temp_vol)

ste_inv_temp_vol = std_inv_temp_vol/np.sqrt(amount_of_sim)
ste_inv_temp_vol_mean = std_inv_temp_vol_mean/np.sqrt(amount_of_sim)
ste_inv_temp_vol_std = std_inv_temp_vol_std/np.sqrt(amount_of_sim)
ste_inv_temp_reward_vol = std_inv_temp_reward_vol/np.sqrt(amount_of_sim)

#for time plot:
r_vol_inv_temp_cumsum_end = np.divide(r_inv_temp_vol_cumsum, amount_of_sim)
r_vol_inv_temp_end = np.divide(r_inv_temp_vol, amount_of_sim)
invtemperature_vol_end= np.divide(invtemperature_vol, amount_of_sim)
invtemperature_mean_vol_end = np.divide(invtemperature_mean_vol, amount_of_sim)
invtemperature_std_vol_end = np.divide(invtemperature_std_vol, amount_of_sim)

#for time plots of untransformed ML
MLinv_temp_vol_end = np.divide(MLinv_temp_vol, amount_of_sim)
MLinv_temp_mean_vol_end = np.divide(MLinv_temp_mean_vol, amount_of_sim)
MLinv_temp_std_vol_end = np.divide(MLinv_temp_std_vol, amount_of_sim)


'''




#################################################################
#second sim
#################################################################

mean_start = 100 #9000 #trial from which mean invtemperature is taken
mean_end = 100 #1000
# META-LEARNING OF INVTEMPERATURE IN A VARIABLE CONTEXT (VARIABILITY IS REINFORCED)
#for barplots:
total_inv_temp_var = np.zeros(amount_of_sim)
total_inv_temp_var_mean = np.zeros(amount_of_sim)
total_inv_temp_var_std = np.zeros(amount_of_sim)
reward_var = np.zeros(amount_of_sim)
reward_var2 = np.zeros(amount_of_sim)

#for time plots:
r_var_cumsum = np.zeros(T)
r_var = np.zeros(T)
e_var = np.zeros(T)
e_mean_var = np.zeros(T)
e_std_var = np.zeros(T)

#for time plots of untransformed ML
ML_var = np.zeros(T)
ML_mean_var = np.zeros(T)
ML_std_var = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_var, inv_temp_var_mean, inv_temp_var_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_adversarial(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_inv_temp_var[sim] = np.mean(inv_temp_var[mean_start:])
    total_inv_temp_var_mean[sim] = np.mean(inv_temp_var_mean[mean_start:])
    total_inv_temp_var_std[sim] = np.mean(inv_temp_var_std[mean_start:])
    reward_var[sim] = np.mean(r[mean_start:])
    reward_var2[sim] = np.mean(r[:mean_end])

    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_var_cumsum = r_var_cumsum + r_cumsum_av
    r_var = r_var + r
    e_var = e_var + inv_temp_var
    e_mean_var = e_mean_var + inv_temp_var_mean
    e_std_var = e_std_var + inv_temp_var_std

    #for time plots of untransformed ML
    ML_var = ML_var + ML_stored
    ML_mean_var = ML_mean_var + ML_mean_stored
    ML_std_var = ML_std_var + ML_std_stored

#for bar plot:
av_inv_temp_var = np.mean(total_inv_temp_var)
av_inv_temp_var_mean = np.mean(total_inv_temp_var_mean)
av_inv_temp_var_std = np.mean(total_inv_temp_var_std)
av_reward_var = np.mean(reward_var)
av_reward_var2 = np.mean(reward_var2)

std_inv_temp_var = np.std(total_inv_temp_var)
std_inv_temp_var_mean = np.std(total_inv_temp_var_mean)
std_inv_temp_var_std = np.std(total_inv_temp_var_std)
std_reward_var = np.std(reward_var)
std_reward_var2 = np.std(reward_var2)

#for time plot:
r_var_cumsum_end = np.divide(r_var_cumsum, amount_of_sim)
r_var_end = np.divide(r_var, amount_of_sim)
e_var_end= np.divide(e_var, amount_of_sim)
e_mean_var_end = np.divide(e_mean_var, amount_of_sim)
e_std_var_end = np.divide(e_std_var, amount_of_sim)

#for time plots of untransformed ML
ML_var_end = np.divide(ML_var, amount_of_sim)
ML_mean_var_end = np.divide(ML_mean_var, amount_of_sim)
ML_std_var_end = np.divide(ML_std_var, amount_of_sim)




# META-LEARNING OF INVTEMPERATURE IN A STABLE CONTEXT (REWARD PROBABILITIES ARE CONSTANT)
#for bar plot:
total_inv_temp_sta = np.zeros(amount_of_sim)
total_inv_temp_sta_mean = np.zeros(amount_of_sim)
total_inv_temp_sta_std = np.zeros(amount_of_sim)
reward_sta = np.zeros(amount_of_sim)
reward_sta2 = np.zeros(amount_of_sim)

#for time plots:
r_sta_cumsum = np.zeros(T)
r_sta = np.zeros(T)
e_sta = np.zeros(T)
e_mean_sta = np.zeros(T)
e_std_sta = np.zeros(T)

#for time plots of untransformed ML
ML_sta = np.zeros(T)
ML_mean_sta = np.zeros(T)
ML_std_sta = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_sta, inv_temp_sta_mean, inv_temp_sta_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_stable(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)

    #for bar plot:
    total_inv_temp_sta[sim] = np.mean(inv_temp_sta[mean_start:])
    total_inv_temp_sta_mean[sim] = np.mean(inv_temp_sta_mean[mean_start:])
    total_inv_temp_sta_std[sim] = np.mean(inv_temp_sta_std[mean_start:])
    reward_sta[sim] = np.mean(r[mean_start:])
    reward_sta2[sim] = np.mean(r[:mean_end])

    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_sta_cumsum = r_sta_cumsum + r_cumsum_av
    r_sta = r_sta + r
    e_sta = e_sta + inv_temp_sta
    e_mean_sta = e_mean_sta + inv_temp_sta_mean
    e_std_sta = e_std_sta + inv_temp_sta_std

    #for time plots of untransformed ML
    ML_sta = ML_sta + ML_stored
    ML_mean_sta = ML_mean_sta + ML_mean_stored
    ML_std_sta = ML_std_sta + ML_std_stored

#for bar plot:
av_inv_temp_sta = np.mean(total_inv_temp_sta)
av_inv_temp_sta_mean = np.mean(total_inv_temp_sta_mean)
av_inv_temp_sta_std = np.mean(total_inv_temp_sta_std)
av_reward_sta = np.mean(reward_sta)
av_reward_sta2 = np.mean(reward_sta2)

std_inv_temp_sta = np.std(total_inv_temp_sta)
std_inv_temp_sta_mean = np.std(total_inv_temp_sta_mean)
std_inv_temp_sta_std = np.std(total_inv_temp_sta_std)
std_reward_sta = np.std(reward_sta)
std_reward_sta2 = np.std(reward_sta2)

#for time plots:
r_sta_cumsum_end = np.divide(r_sta_cumsum, amount_of_sim)
r_sta_end = np.divide(r_sta, amount_of_sim)
e_sta_end= np.divide(e_sta, amount_of_sim)
e_mean_sta_end = np.divide(e_mean_sta, amount_of_sim)
e_std_sta_end = np.divide(e_std_sta, amount_of_sim)

#for time plots of untransformed ML
ML_sta_end = np.divide(ML_sta, amount_of_sim)
ML_mean_sta_end = np.divide(ML_mean_sta, amount_of_sim)
ML_std_sta_end = np.divide(ML_std_sta, amount_of_sim)





# META-LEARNING OF INVTEMPERATURE IN A VOLATILE CONTEXT (REWARD PROBABILITIES SHUFFLE)
#for bar plot:
total_inv_temp_vol = np.zeros(amount_of_sim)
total_inv_temp_vol_mean = np.zeros(amount_of_sim)
total_inv_temp_vol_std = np.zeros(amount_of_sim)
reward_vol = np.zeros(amount_of_sim)
reward_vol2 = np.zeros(amount_of_sim)

#for time plots:
r_vol_cumsum = np.zeros(T)
r_vol = np.zeros(T)
e_vol = np.zeros(T)
e_mean_vol = np.zeros(T)
e_std_vol = np.zeros(T)

#for time plots of untransformed ML
ML_vol = np.zeros(T)
ML_mean_vol = np.zeros(T)
ML_std_vol = np.zeros(T)

for sim in range(amount_of_sim):
    k, r, Q_k_stored, inv_temp_vol, inv_temp_vol_mean, inv_temp_vol_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLinv_temp_volatile(x=x, Q_alpha=Q_alpha, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLinv_temp_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_inv_temp_vol[sim] = np.mean(inv_temp_vol[mean_start:])
    total_inv_temp_vol_mean[sim] = np.mean(inv_temp_vol_mean[mean_start:])
    total_inv_temp_vol_std[sim] = np.mean(inv_temp_vol_std[mean_start:])
    reward_vol[sim] = np.mean(r[mean_start:])
    reward_vol2[sim] = np.mean(r[:mean_end])

    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_vol_cumsum = r_vol_cumsum + r_cumsum_av
    r_vol = r_vol + r
    e_vol = e_vol + inv_temp_vol
    e_mean_vol = e_mean_vol + inv_temp_vol_mean
    e_std_vol = e_std_vol + inv_temp_vol_std

    #for time plots of untransformed ML
    ML_vol = ML_vol + ML_stored
    ML_mean_vol = ML_mean_vol + ML_mean_stored
    ML_std_vol = ML_std_vol + ML_std_stored

#for bar plot:
av_inv_temp_vol = np.mean(total_inv_temp_vol)
av_inv_temp_vol_mean = np.mean(total_inv_temp_vol_mean)
av_inv_temp_vol_std = np.mean(total_inv_temp_vol_std)
av_reward_vol = np.mean(reward_vol)
av_reward_vol2 = np.mean(reward_vol2)

std_inv_temp_vol = np.std(total_inv_temp_vol)
std_inv_temp_vol_mean = np.std(total_inv_temp_vol_mean)
std_inv_temp_vol_std = np.std(total_inv_temp_vol_std)
std_reward_vol = np.std(reward_vol)
std_reward_vol2 = np.std(reward_vol2)

#for time plots:
r_vol_cumsum_end = np.divide(r_vol_cumsum, amount_of_sim)
r_vol_end = np.divide(r_vol, amount_of_sim)
e_vol_end= np.divide(e_vol, amount_of_sim)
e_mean_vol_end = np.divide(e_mean_vol, amount_of_sim)
e_std_vol_end = np.divide(e_std_vol, amount_of_sim)

#for time plots of untransformed ML
ML_vol_end = np.divide(ML_vol, amount_of_sim)
ML_mean_vol_end = np.divide(ML_mean_vol, amount_of_sim)
ML_std_vol_end = np.divide(ML_std_vol, amount_of_sim)


'''
#################################################################
#PLOTTING
#################################################################

save_dir_first = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/inv_temp-8choice'
new_sim_folder = f'sim{sim_nr}'
save_dir = os.path.join(save_dir_first, new_sim_folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

time = np.linspace(1, T, T, endpoint=True)
#Parameter evolution
fig_name = os.path.join(save_dir, f'sim{sim_nr}_param_evolution')

#time plot for reward:
f4, ax10 = plt.subplots(1, 1, figsize=(12,15))

adversarial, = ax10.plot(time, invtemperature_var_end, label=f'adversarial context')
stable, = ax10.plot(time, invtemperature_sta_end, label=f'stable context')
volatile, = ax10.plot(time, invtemperature_vol, label=f'volatile context')

ax10.set_title('evolution of invtemperature in different contexts')
ax10.set_xlabel('trials')
ax10.set_ylabel('invtemperature')
ax10.legend(handles=[adversarial, stable, volatile])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_invtemperature_ifo_time')
f4.suptitle(f'rewards for each model averaged over {amount_of_sim} simulations')
plt.savefig(fig_name)
plt.show()

store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'learning rate for Q-value' : Q_alpha,
    'learning rate for the mean of the meta-learning parameter' : ML_alpha_mean,
    'learning rate for the std of the meta-learning parameter' : ML_alpha_std,
    'initial mean invtemperature value' : inv_temp_mean_int,
    'initial standard deviation of meta-learning parameter in logit transform' : ML_std_int,
    'reward probabilities in stable context' : reward_st,
    'reward probabiities in volatile context' : reward_vl
     }

title_excel = os.path.join(save_dir, f'sim{sim_nr}_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)
'''

#################################################################
#second PLOTTING
#################################################################

save_dir_first = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/inv_temp-8choice'
new_sim_folder = f'sim{sim_nr}'
save_dir = os.path.join(save_dir_first, new_sim_folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#bar plots:
title1 = f'average sampled invtemperature over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha} and ML learning rate of {ML_alpha_mean}\naverage gained rewards in last 1000 trials are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}\naverage gained rewards in first 1000 trials are for stable {round(av_reward_sta2, 2)} +/- {round(std_reward_sta2,2)}, for volatile {round(av_reward_vol2,2)} +/- {round(std_reward_vol2,2)} and for variable {round(av_reward_var2,2)} +/- {round(std_reward_var2,2)}'
fig_name = os.path.join(save_dir, f'sim{sim_nr}_sampled_invtemperature')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_inv_temp_sta, av_inv_temp_vol, av_inv_temp_var], yerr=[std_inv_temp_sta, std_inv_temp_vol, std_inv_temp_var])
ax.set_ylabel('invtemperature')
plt.title(title1)
plt.savefig(fig_name)
#plt.show()

store_av_inv_temp = {
    'av_LR_var' : av_inv_temp_var,
    'std_LR_var' : std_inv_temp_var,
    'av_LR_sta' : av_inv_temp_sta,
    'std_LR_sta' : std_inv_temp_sta,
    'av_LR_vol' : av_inv_temp_vol,
    'std_LR_vol' : std_inv_temp_vol,

}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_inv_temp{inv_temp_mean_int}.xlsx')
df = pd.DataFrame(data=store_av_inv_temp, index=[1])
df.to_excel(title_excel, index=False)

title2 = f'average updated mean invtemperature over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha} and ML learning rate of {ML_alpha_mean}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'sim{sim_nr}_mean_invtemperature')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_inv_temp_sta_mean, av_inv_temp_vol_mean, av_inv_temp_var_mean], yerr=[std_inv_temp_sta_mean, std_inv_temp_vol_mean, std_inv_temp_var_mean])
ax.set_ylabel('invtemperature')
plt.title(title2) 
plt.savefig(fig_name)
#plt.show()

#time plots:

time = np.linspace(1, T, T, endpoint=True)

#time plot for invtemperature:
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,15))

evar, = ax1.plot(time, e_var_end, label=f'sampled invtemperature')
evar_mean, = ax1.plot(time, e_mean_var_end, label=f'mean invtemperature')
evar_std, = ax1.plot(time, e_std_var_end, label=f'standard deviation of invtemperature')

esta, = ax2.plot(time, e_sta_end, label=f'sampled invtemperature')
esta_mean, = ax2.plot(time, e_mean_sta_end, label=f'mean invtemperature')
esta_std, = ax2.plot(time, e_std_sta_end, label=f'standard deviation of invtemperature')

evol, = ax3.plot(time, e_vol_end, label=f'sampled invtemperature')
evol_mean, = ax3.plot(time, e_mean_vol_end, label=f'mean invtemperature')
evol_std, = ax3.plot(time, e_std_vol_end, label=f'standard deviation of invtemperature')

ax1.set_title('variable context')
ax1.set_xlabel('trials')
ax1.set_ylabel('invtemperature')
ax1.legend(handles=[evar, evar_mean, evar_std])

ax2.set_title('stable context')
ax2.set_xlabel('trials')
ax2.set_ylabel('invtemperature')
ax2.legend(handles=[esta, esta_mean, esta_std])

ax3.set_title('volatile context')
ax3.set_xlabel('trials')
ax3.set_ylabel('invtemperature')
ax3.legend(handles=[evol, evol_mean, evol_std])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_inv_temp_ifo_time')
f.suptitle(f'invtemperature averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#time plot for untransformed ML:
f2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12,15))

MLvar, = ax4.plot(time, ML_var_end, label=f'sampled ML')
MLvar_mean, = ax4.plot(time, ML_mean_var_end, label=f'mean ML')
MLvar_std, = ax4.plot(time, ML_std_var_end, label=f'standard deviation of ML')

MLsta, = ax5.plot(time, ML_sta_end, label=f'sampled ML')
MLsta_mean, = ax5.plot(time, ML_mean_sta_end, label=f'mean ML')
MLsta_std, = ax5.plot(time, ML_std_sta_end, label=f'standard deviation of ML')

MLvol, = ax6.plot(time, ML_vol_end, label=f'sampled ML')
MLvol_mean, = ax6.plot(time, ML_mean_vol_end, label=f'mean ML')
MLvol_std, = ax6.plot(time, ML_std_vol_end, label=f'standard deviation of ML')

ax4.set_title('variable context')
ax4.set_xlabel('trials')
ax4.set_ylabel('ML before transforming to inv_temp')
ax4.legend(handles=[MLvar, MLvar_mean, MLvar_std])

ax5.set_title('stable context')
ax5.set_xlabel('trials')
ax5.set_ylabel('ML before transforming to inv_temp')
ax5.legend(handles=[MLsta, MLsta_mean, MLsta_std])

ax6.set_title('volatile context')
ax6.set_xlabel('trials')
ax6.set_ylabel('ML before transforming to inv_temp')
ax6.legend(handles=[MLvol, MLvol_mean, MLvol_std])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_ML_ifo_time')
f2.suptitle(f'untransformed meta-learning (ML) parameter averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#time plot for untransformed ML vs transformed inv_temp:
f3, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(12,15))

var1, = ax7.plot(time, ML_var_end, label=f'sampled ML')
var2, = ax7.plot(time, ML_mean_var_end, label=f'mean ML')
var3, = ax7.plot(time, e_var_end, label=f'sampled inv_temp (transformed ML)')
var4, = ax7.plot(time, e_mean_var_end, label=f'mean inv_temp (transformed ML)')

sta1, = ax8.plot(time, ML_sta_end, label=f'sampled ML')
sta2, = ax8.plot(time, ML_mean_sta_end, label=f'mean ML')
sta3, = ax8.plot(time, e_sta_end, label=f'sampled inv_temp (transformed ML)')
sta4, = ax8.plot(time, e_mean_sta_end, label=f'mean inv_temp (transformed ML)')

vol1, = ax9.plot(time, ML_vol_end, label=f'sampled ML')
vol2, = ax9.plot(time, ML_mean_vol_end, label=f'mean ML')
vol3, = ax9.plot(time, e_vol_end, label=f'sampled inv_temp (transformed ML)')
vol4, = ax9.plot(time, e_mean_vol_end, label=f'mean inv_temp (transformed ML)')

ax7.set_title('variable context')
ax7.set_xlabel('trials')
ax7.set_ylabel('ML/inv_temp')
ax7.legend(handles=[var1, var2, var3, var4])

ax8.set_title('stable context')
ax8.set_xlabel('trials')
ax8.set_ylabel('ML/inv_temp')
ax8.legend(handles=[sta1, sta2, sta3, sta4])

ax9.set_title('volatile context')
ax9.set_xlabel('trials')
ax9.set_ylabel('ML/inv_temp')
ax9.legend(handles=[vol1, vol2, vol3, vol4])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_MLvsInv_temp_ifo_time')
f2.suptitle(f'untransformed meta-learning (ML) parameter and transformed invtemperature averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()


#time plot for reward:
f4, (ax10, ax11) = plt.subplots(2, 1, figsize=(12,15))

Rvar, = ax10.plot(time, r_var_end, label=f'reward in variable context')
Rsta, = ax10.plot(time, r_sta_end, label=f'reward in stable context')
Rvol, = ax10.plot(time, r_vol_end, label=f'reward in volatile context')

Rsum_var, = ax11.plot(time, r_var_cumsum_end, label=f'cumulative reward in variable context')
Rsum_sta, = ax11.plot(time, r_sta_cumsum_end, label=f'cumulative reward in stable context')
Rsum_vol, = ax11.plot(time, r_vol_cumsum_end, label=f'cumulative reward in volatile context')

ax10.set_title('reward in funcion of time')
ax10.set_xlabel('trials')
ax10.set_ylabel('Reward')
ax10.legend(handles=[Rvar, Rsta, Rvol])

ax11.set_title('cumulative reward in function of time')
ax11.set_xlabel('trials')
ax11.set_ylabel('cumulative reward')
ax11.legend(handles=[Rsum_var, Rsum_sta, Rsum_vol])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_rewards_ifo_time')
f4.suptitle(f'rewards in each context averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#conference paper figure:
fig_name = os.path.join(save_dir, f'sim{sim_nr}_conference_plot')
fig, ax12 = plt.subplots(figsize=(6, 3))
invtemperaturesta, = ax12.plot(time, e_sta_end, label=f'stable environment', color = 'darkcyan')
invtemperaturevol, = ax12.plot(time, e_vol_end, label=f'volatile environment', color = 'darkorange')
invtemperaturevar, = ax12.plot(time, e_var_end, label=f'variable environment', color = 'forestgreen')
ax12.legend(handles=[invtemperaturesta, invtemperaturevol, invtemperaturevar], loc='center left', bbox_to_anchor=(1, 0.5))
ax12.set_xlabel('trials', fontsize=16)
ax12.set_ylabel('invtemperature', fontsize=16)
plt.ylim([0, 20])
plt.xlim([0, T])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.legend(handles=[invtemperaturesta,invtemperaturevol,invtemperaturevar])
#ax12.set_title(f'meta-learning of invtemperature, based on {amount_of_sim} simulations')
plt.savefig(fig_name)
plt.show()

global_mean_reward = (av_reward_sta + av_reward_var + av_reward_vol)/3
global_ste_reward = (np.sqrt((std_reward_var**2)+(std_reward_vol**2)+(std_reward_sta**2)))/np.sqrt(amount_of_sim*3)

store_av_reward = {
    'av_reward_var' : av_reward_var,
    'std_reward_var' : std_reward_var,
    'av_reward_sta' : av_reward_sta,
    'std_reward_sta' : std_reward_sta,
    'av_reward_vol' : av_reward_vol,
    'std_reward_vol' : std_reward_vol,
    'global_mean_reward' : global_mean_reward,
    'global_ste_reward' : global_ste_reward
}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_rewards_inv_temp{inv_temp_mean_int}.xlsx')
df = pd.DataFrame(data=store_av_reward, index=[1])
df.to_excel(title_excel, index=False)




store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'learning rate for Q-value' : Q_alpha,
    'learning rate for the mean of the meta-learning parameter' : ML_alpha_mean,
    'learning rate for the std of the meta-learning parameter' : ML_alpha_std,
    'initial mean invtemperature value' : inv_temp_mean_int,
    'initial standard deviation of meta-learning parameter in logit transform' : ML_std_int,
    'reward probabilities in stable context' : reward_st,
    'reward probabiities in volatile context' : reward_vl
     }

title_excel = os.path.join(save_dir, f'sim{sim_nr}a_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)

