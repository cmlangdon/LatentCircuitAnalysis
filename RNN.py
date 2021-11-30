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
    def __init__(self, connectivity, mask, n, radius=1.5,lambda_r=0, lambda_o=0, tau=200, sigma_rec=.15):
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
        self.activation = torch.nn.ReLU()

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
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states,self.output_layer(states)[:, self.mask, :], self.output_layer(states)


class RNNNet(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_loss(self, y_pred, y_true, X=None, training=False):

        y = to_tensor(y_true, device=self.device)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        return self.criterion_(y, self.module_.output_layer(y_pred[0])[:, self.module_.mask, :]) + \
               self.module_.lambda_r * torch.mean(torch.pow(y_pred[0], 2)) + \
               self.module_.lambda_o * L2_ortho(self)

    def train_step(self, Xi, yi, **fit_params):
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)

        # Enforce Dale's Law after optimizer step
        if self.module_.dale:
            self.module_.recurrent_layer.weight.data = torch.relu(
                self.module_.recurrent_layer.weight.data * self.module_.dale_mask) * self.module_.dale_mask

        # Enforce positive entries for input and output layers after optimizer step
        self.module_.input_layer.weight.data = self.module_.input_mask * torch.relu(self.module_.input_layer.weight.data)
        self.module_.output_layer.weight.data = self.module_.output_mask * torch.relu(self.module_.output_layer.weight.data)


        return step_accumulator.get_step()


# Define some epoch scorers for monitoring progress during training.
def R2_task(net, X, y):
    y_true = to_tensor(y,device=device)
    x = to_tensor(net.predict(X), device=device)
    z = net.module_.output_layer(x)[:, net.module_.mask, :]
    var_y = torch.var(y_true, unbiased=False)
    return 1-F.mse_loss(y_true, z,reduction='mean') / var_y


def L2_rate(net, X, y):
    x = to_tensor(net.predict(X), device=device)
    return torch.mean(torch.pow(x, 2))


def L2_ortho(net,X = None, y = None):
    b = torch.cat((net.module_.input_layer.weight, net.module_.output_layer.weight.t()), dim=1)
    b = b / torch.norm(b, dim=0)
    return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)
