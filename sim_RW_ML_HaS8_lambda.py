#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of lambda_unchosen (addition) with policy-gradient method
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    adversarial: least frequent 60% of sequences
                         
Critical parameters: epsilon, learning rate, unchosen addition (there is no chosen here)


@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random


#simulation of Rescorla-Wagner model with meta-learning of unchosen option value 
#meta-learning goes through parameter ML which is transformed to unchosen with a logit transformation
#rewards are baselined
def simulate_RW_MLuc_adversarial(x, Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #scale              --->        scale with which ML-parameter gets transformed to unchosen in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
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
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(uc_mean_int/(1-uc_mean_int))
    #ML_std_int = (1/scale)*np.log(uc_std_int/(1-uc_std_int))

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
        #unchosen is calculated with a logit transformation of the ML
        uc = ML #np.exp(scale*ML)/(1+np.exp(scale*ML))
        uc_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        uc_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

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
            #ML_mean = np.min([ML_mean,10])
            #ML_mean = np.max([ML_mean, -10])
            #ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored

#simulation of Rescorla-Wagner model with meta-learning of unchosen option value 
#meta-learning goes through parameter ML which is transformed to unchosen with a logit transformation
#rewards are baselined
def simulate_RW_MLuc_stable(x, Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #scale              --->        scale with which ML-parameter gets transformed to unchosen in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
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
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(uc_mean_int/(1-uc_mean_int))
    #ML_std_int = (1/scale)*np.log(uc_std_int/(1-uc_std_int))

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
        #unchosen is calculated with a logit transformation of the ML
        uc = ML # np.exp(scale*ML)/(1+np.exp(scale*ML))
        uc_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        uc_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))


        #stable
        a1 = reward_stable[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])
        
        
        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

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
            #ML_mean = np.min([ML_mean,10])
            #ML_mean = np.max([ML_mean, -10])
            #ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored


#simulation of Rescorla-Wagner model with meta-learning of unchosen option value 
#meta-learning goes through parameter ML which is transformed to unchosen with a logit transformation
#rewards are baselined
def simulate_RW_MLuc_volatile(x, Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable, reward_volatile):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #scale              --->        scale with which ML-parameter gets transformed to unchosen in a logit transformation
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #percentile         --->        
    #reward_stable      --->
    #reward_volatile    --->
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
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
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(uc_mean_int/(1-uc_mean_int))
    #ML_std_int = (1/scale)*np.log(uc_std_int/(1-uc_std_int))

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
        #unchosen is calculated with a logit transformation of the ML
        uc = ML # np.exp(scale*ML)/(1+np.exp(scale*ML))
        uc_mean = ML_mean #np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        uc_std = ML_std #np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

        
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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

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
            #ML_mean = np.min([ML_mean,10])
            #ML_mean = np.max([ML_mean, -10])
            #ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored

sim_nr = 'fwo2'
amount_of_sim = 500
x = 10000
T = 1*x
K = 8

reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_st = '3-5 70-30'
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
reward_vl = '3-5 90-10'

Q_alpha = 0.5
eps = 0.5
unchosen = 0


eps_mean_int = 0.5
MLeps_mean_int = np.log(eps_mean_int/(1-eps_mean_int))
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



mean_start = 9000

#################################################################
#SIMULATIONS
#################################################################

#variable context:
#for time plots:
r_var_cumsum = np.zeros(T)
r_var = np.zeros(T)

g_var = np.zeros(T)
g_mean_var = np.zeros(T)
g_std_var = np.zeros(T)

l_var = np.zeros(T)
l_mean_var = np.zeros(T)
l_std_var = np.zeros(T)

total_uc_var = np.zeros(amount_of_sim)

reward_var = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, unchosen_var, unchosen_var_mean, unchosen_var_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = simulate_RW_MLuc_adversarial(x=x, Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)

    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_var[sim] = np.mean(r[mean_start:])
    total_uc_var[sim] = np.mean(unchosen_var[mean_start:])

    r_var_cumsum = r_var_cumsum + r_cumsum_av
    r_var = r_var + r

    g_var = g_var + unchosen_var
    g_mean_var = g_mean_var + unchosen_var_mean
    g_std_var = g_std_var + unchosen_var_std




#for average:
av_uc_var = np.mean(total_uc_var)
std_uc_var = np.std(total_uc_var)

#for time plot:
r_var_cumsum_end = np.divide(r_var_cumsum, amount_of_sim)
r_var_end = np.divide(r_var, amount_of_sim)

g_var_end= np.divide(g_var, amount_of_sim)
g_mean_var_end = np.divide(g_mean_var, amount_of_sim)
g_std_var_end = np.divide(g_std_var, amount_of_sim)

av_reward_var = np.mean(reward_var)
std_reward_var = np.std(reward_var)

#stable context:
#for time plots:
r_sta_cumsum = np.zeros(T)
r_sta = np.zeros(T)

g_sta = np.zeros(T)
g_mean_sta = np.zeros(T)
g_std_sta = np.zeros(T)


total_uc_sta = np.zeros(amount_of_sim)

reward_sta = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    
    k, r, Q_k_stored, unchosen_sta, unchosen_sta_mean, unchosen_sta_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = simulate_RW_MLuc_stable(x=x, Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)

    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_sta[sim] = np.mean(r[mean_start:])

    total_uc_sta[sim] = np.mean(unchosen_sta[mean_start:])

    r_sta_cumsum = r_sta_cumsum + r_cumsum_av
    r_sta = r_sta + r

    g_sta = g_sta + unchosen_sta
    g_mean_sta = g_mean_sta + unchosen_sta_mean
    g_std_sta = g_std_sta + unchosen_sta_std



#for average:
av_uc_sta = np.mean(total_uc_sta)
std_uc_sta = np.std(total_uc_sta)

#for time plot:
r_sta_cumsum_end = np.divide(r_sta_cumsum, amount_of_sim)
r_sta_end = np.divide(r_sta, amount_of_sim)

g_sta_end= np.divide(g_sta, amount_of_sim)
g_mean_sta_end = np.divide(g_mean_sta, amount_of_sim)
g_std_sta_end = np.divide(g_std_sta, amount_of_sim)

av_reward_sta = np.mean(reward_sta)
std_reward_sta = np.std(reward_sta)

#volatile context:
#for time plots:
r_vol_cumsum = np.zeros(T)
r_vol = np.zeros(T)

g_vol = np.zeros(T)
g_mean_vol = np.zeros(T)
g_std_vol = np.zeros(T)


total_uc_vol = np.zeros(amount_of_sim)

reward_vol = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    
    k, r, Q_k_stored, unchosen_vol, unchosen_vol_mean, unchosen_vol_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = simulate_RW_MLuc_volatile(x=x, Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, reward_stable=reward_stable, reward_volatile=reward_volatile)

    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_vol[sim] = np.mean(r[mean_start:])
    total_uc_vol[sim] = np.mean(unchosen_vol[mean_start:])

    r_vol_cumsum = r_vol_cumsum + r_cumsum_av
    r_vol = r_vol + r

    g_vol = g_vol + unchosen_vol
    g_mean_vol = g_mean_vol + unchosen_vol_mean
    g_std_vol = g_std_vol + unchosen_vol_std



#for average:
av_uc_vol = np.mean(total_uc_vol)
std_uc_vol = np.std(total_uc_vol)


#for time plot:
r_vol_cumsum_end = np.divide(r_vol_cumsum, amount_of_sim)
r_vol_end = np.divide(r_vol, amount_of_sim)

g_vol_end= np.divide(g_vol, amount_of_sim)
g_mean_vol_end = np.divide(g_mean_vol, amount_of_sim)
g_std_vol_end = np.divide(g_std_vol, amount_of_sim)

av_reward_vol = np.mean(reward_vol)
std_reward_vol = np.std(reward_vol)

#################################################################
#PLOTTING
#################################################################

save_dir_first = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/lambda_unchosen-8choice'
new_sim_folder = f'sim{sim_nr}'
save_dir = os.path.join(save_dir_first, new_sim_folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


store_av_eps = {
    'av_uc_var' : av_uc_var,
    'std_uc_var' : std_uc_var,
    'av_uc_sta' : av_uc_sta,
    'std_uc_sta' : std_uc_sta,
    'av_uc_vol' : av_uc_vol,
    'std_uc_vol' : std_uc_vol,


}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_c_uc.xlsx')
df = pd.DataFrame(data=store_av_eps, index=[1])
df.to_excel(title_excel, index=False)

#time plots:

time = np.linspace(1, T, T, endpoint=True)

fig_name = os.path.join(save_dir, f'sim{sim_nr}_unchosen-chosen_compare_contexts')
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10, 21))
unchosensta, = ax1.plot(time, g_sta_end, label=f'unchosen (unchosen options) stable environment')
unchosenvol, = ax2.plot(time, g_vol_end, label=f'unchosen (unchosen options) volatile environment')
unchosenvar, = ax3.plot(time, g_var_end, label=f'unchosen (unchosen options) hypervolatile environment')


ax1.legend(handles=[unchosensta])
ax2.legend(handles=[unchosenvol])
ax3.legend(handles=[unchosenvar])

ax1.set_xlabel('trials')
ax1.set_ylabel('unchosen')
ax2.set_xlabel('trials')
ax2.set_ylabel('unchosen')
ax3.set_xlabel('trials')
ax3.set_ylabel('unchosen')

#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)
#plt.show()

f2, (ax4, ax5) = plt.subplots(2, 1, figsize=(10,14))
reward_sta_cum, = ax4.plot(time, r_sta_cumsum_end, label=f'stable context')
reward_sta, = ax5.plot(time, r_sta_end, label=f'stable context')

reward_vol_cum, = ax4.plot(time, r_vol_cumsum_end, label=f'volatile context')
reward_vol, = ax5.plot(time, r_vol_end, label=f'volatile context')

reward_var_cum, = ax4.plot(time, r_var_cumsum_end, label=f'variable context')
reward_var, = ax5.plot(time, r_var_end, label=f'variable context')

ax4.legend(handles=[reward_sta_cum, reward_vol_cum, reward_var_cum])
ax5.legend(handles=[reward_sta, reward_vol, reward_var])
fig_name = os.path.join(save_dir, f'sim{sim_nr}_reward_compare_contexts')
ax2.set_xlabel('trials')
ax2.set_xlabel('cumulative reward')
ax3.set_xlabel('trials')
ax3.set_ylabel('reward')
plt.savefig(fig_name)
#plt.show()

#conference paper figure:
fig_name = os.path.join(save_dir, f'sim{sim_nr}conference_plot')
fig, ax12 = plt.subplots(figsize=(6, 3))
unchosensta, = ax12.plot(time, g_sta_end, label=f'stable environment', color = 'darkcyan')
unchosenvol, = ax12.plot(time, g_vol_end, label=f'volatile environment', color='darkorange')
unchosenvar, = ax12.plot(time, g_var_end, label=f'adversarial environment', color='forestgreen')
ax12.legend(handles=[unchosensta, unchosenvol, unchosenvar], loc='center left', bbox_to_anchor=(1, 0.5))
ax12.set_xlabel('trials', fontsize=16)
ax12.set_ylabel('lambda', fontsize=16)
plt.xlim([0, 10000])
plt.ylim([-3,3])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.set_title(f'meta-learning of learning rate, based on {amount_of_sim} simulations')
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


title_excel = os.path.join(save_dir, f'sim{sim_nr}av_rewards_lambda_unchosen.xlsx')
df = pd.DataFrame(data=store_av_reward, index=[1])
df.to_excel(title_excel, index=False)




K=len(reward_stable)

store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'learning rate for Q-value' : Q_alpha,
    'epsilon' : eps,
    'learning rate for the mean of the unchosen meta-learning parameter' : ML_alpha_mean,
    'learning rate for the std of the unchosen meta-learning parameter' : ML_alpha_std,
    'initial mean unchosen value' : MLuc_mean_int,
    'initial standard deviation of unchosen value' : ML_std_int,
    'reward probabilities in stable context' : reward_st,
    'reward probabiities in volatile context' : reward_vl    }

title_excel = os.path.join(save_dir, f'sim{sim_nr}a_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)

             