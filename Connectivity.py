#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:15:39 2020

@author: langdon
"""

import numpy as np
import torch
from scipy.sparse import random
from scipy import stats
from numpy import linalg

    
def large_connectivity(device,N,radius=1.5,recurrent_sparsity=1,input_sparsity=1,output_sparsity=1):
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)
    N_in=6
    N_out=2
    # Initialize W_rec
    W_rec = torch.empty([0,N])
    
    # Balancing parameters
    mu_E = 1/np.sqrt(N)
    mu_I = 4/np.sqrt(N)
    
    var = 1/N
 
    rowE = torch.empty([Ne,0])
    rowI = torch.empty([Ni,0])      
    
    rowE = torch.cat((rowE,torch.tensor(random(Ne, Ne, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()),1)
    rowE = torch.cat((rowE,-torch.tensor(random(Ne, Ni, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_I).rvs).toarray()).float()),1)
    rowI = torch.cat((rowI,torch.tensor(random(Ni,Ne, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()),1)
    rowI = torch.cat((rowI,-torch.tensor(random(Ni, Ni, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_I).rvs).toarray()).float()),1)
   
    W_rec = torch.cat((W_rec,rowE),0)
    W_rec = torch.cat((W_rec,rowI),0)

    W_rec= W_rec-torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius*W_rec/spec_radius
    
    W_in = torch.zeros([N,N_in]).float()
    W_in[:,:] = torch.tensor(random(N, N_in, density=input_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()
    
    W_out = torch.zeros([N_out,N])
    W_out[:,:Ne]=torch.tensor(random(N_out, Ne, density=output_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()

    dale_mask = torch.sign(W_rec).to(device=device).float()
    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_in != 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(), W_in.to(device=device).float(), W_out.to(device=device).float(),recurrent_mask.to(device=device).float(), dale_mask, output_mask, input_mask




def small_connectivity(device):

    input_mask = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0)

    output_mask = torch.cat((torch.zeros(2,6),torch.eye(2)),dim=1).float()

    #recurrent_mask = torch.ones(8,8)
    w_rec = torch.eye(8)

    # context mechanism
    w_rec[[4,5],0]=-1
    w_rec[[2,3],1]=-1

    # Output
    w_rec[6,[2,4]]=1
    w_rec[7,[3,5]]=1

    # Off diagonal
    for i in range(4):
        w_rec[2*i+1,2*i]=-1
        w_rec[2*i,2*i+1]=-1

    # stimulus competition    
    #w_rec[4:6,2:4] = -1
    #w_rec[2:4,4:6] = -1

    # Context to choice
    #w_rec[6:8,:2] = 1

    # Stimulus to context
    w_rec[0:2,2:6]=+1

    # Choice to context
    #w_rec[:2,6:]=-1

    # Choice to stimulus
    w_rec[2:6,6:]=-1
    recurrent_mask = w_rec

    return recurrent_mask.to(device=device).float(), input_mask.to(device=device).float(), output_mask.to(device=device).float()
