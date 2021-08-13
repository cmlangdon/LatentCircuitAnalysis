#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:02:27 2020

@author: langdon
"""
from abc import ABC

import torch.nn as nn
from Connectivity import *
from Trials import *

import torch
from sklearn.metrics import make_scorer
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from skorch.callbacks import EarlyStopping
from skorch.callbacks import BatchScoring
from skorch.utils import to_tensor
from torch.nn import functional as F


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class RNNModule(torch.nn.Module):
    def __init__(self, connectivity, mask, n, embedding=False, radius=1.5,lambda_r=0, lambda_o=0, lambda_i=0, lambda_w=0, tau=200, sigma_rec=.15):
        super(RNNModule, self).__init__()
        self.alpha = .2
        self.tau = tau
        self.sigma_rec = torch.tensor(sigma_rec)
        self.connectivity = connectivity
        self.mask = mask
        self.n = n
        self.N = 150
        self.input_size = 6
        self.output_size = 2
        self.radius = radius
        self.lambda_r = lambda_r
        self.lambda_o = lambda_o
        self.lambda_w = lambda_w

        if embedding:
            self.A = torch.rand(self.N, self.N, device=device)
            self.Q = (torch.eye(self.N, device=device) - (self.A - self.A.t()) / 2) @ torch.inverse(
                torch.eye(self.N, device=device) + (self.A - self.A.t()) / 2)
            self.q = self.Q[:self.n, :]
            self.q = self.q.to(device=device)
            self.embedding = self.q
        else:
            self.embedding = torch.eye(self.n)



        if connectivity == 'large':
            self.dale = True
            w_rec, w_in, w_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask = large_connectivity(device, self.n,
                                                                                                       self.radius)

            self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
            self.recurrent_layer.weight.data = w_rec

            self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
            self.input_layer.weight.data = w_in

            self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
            self.output_layer.weight.data = w_out

        else:
            self.dale = False
            self.recurrent_mask, self.input_mask, self.output_mask = small_connectivity(device=device)

            self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
            self.recurrent_layer.weight.data = 0 * self.recurrent_mask * self.recurrent_layer.weight.data.to(device=device)

            self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
            self.input_layer.weight.data = self.input_mask * self.input_layer.weight.data.to(device=device)

            self.output_layer = nn.Linear( self.n, self.output_size, bias=False)
            self.output_layer.weight.data = self.output_mask * self.output_layer.weight.data.to(device=device)


    def forward(self, u):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]
        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                         std=1).to(
            device=device)

        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                torch.relu(self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return self.output_layer(states)[:, self.mask, :], self.output_layer(states),  states


class RNNNet(NeuralNetRegressor):
    def __init__(self, baseline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline= baseline
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y = to_tensor(y_true, device=self.device)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        return L2_task(y, y_pred[0]) + \
                self.module_.lambda_r * L2_rate(y, y_pred[2]) +\
                self.module_.lambda_w * L2_weight(self) +\
                self.module_.lambda_o * L2_ortho(self)

    def train_step(self, Xi, yi, **fit_params):
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)
        if self.module_.dale:
            self.module_.recurrent_layer.weight.data = torch.relu(
                self.module_.recurrent_layer.weight.data * self.module_.dale_mask) * self.module_.dale_mask

        #self.module_.recurrent_layer.weight.data = torch.relu(self.module_.recurrent_layer.weight.data * self.module_.recurrent_mask) * self.module_.recurrent_mask
        self.module_.input_layer.weight.data = self.module_.input_mask * torch.relu(self.module_.input_layer.weight.data)
        self.module_.output_layer.weight.data = self.module_.output_mask * torch.relu(self.module_.output_layer.weight.data)


        return step_accumulator.get_step()

# Task performance
def L2_task(y, y_pred):
    y_pred = to_tensor(y_pred, device=device)
    y = to_tensor(y, device=device)
    criterion = torch.nn.MSELoss()
    return criterion(y_pred, y)

def r2_scorer(y_true, y_pred):
    y_true = to_tensor(y_true,device=device)
    y_pred = to_tensor(y_pred,device=device)
    var_y = torch.var(y_true, unbiased=False)
    return 1-F.mse_loss(y_pred, y_true,reduction='mean') / var_y
r2 = EpochScoring(scoring=make_scorer(r2_scorer), on_train=False)

# Regularizers
def L2_rate(_, y_pred):
    y_pred = to_tensor(y_pred, device=device)
    return torch.mean(torch.pow(y_pred, 2))


def L2_weight(net,X = None, y = None):
    return torch.mean(torch.pow(net.module_.recurrent_layer.weight, 2))


def L2_ortho(net,X = None, y = None):
    b = torch.cat((net.module_.input_layer.weight, net.module_.output_layer.weight.t()), dim=1)
    #b = net.module_.input_layer.weight
    b = b / torch.norm(b, dim=0)
    return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)




# Stopping criteria
def stopping(net,X, y):
    return np.sum(np.abs(np.asarray(net.history[:,'L2_task']) - net.baseline))





