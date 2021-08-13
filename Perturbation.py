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


# Define perturbation class
class Perturbation(nn.Module):

    def __init__(self, tau, sigma_rec, n):
        super(Perturbation, self).__init__()

        self.tau = tau
        self.sigma_rec = torch.tensor(sigma_rec)
        self.n = n
        self.input_size = 6
        self.output_size = 2

        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.requires_grad = False

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.requires_grad = False

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.requires_grad = False

    def forward(self, u, p, alpha):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]
        noise = torch.sqrt(2 * alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                         std=1).to(
            device=device)

        for i in range(t - 1):
            state_new = (1 - alpha) * states[:, i, :] + alpha * (
                torch.relu(
                    self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :] + p[:, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        out = self.output_layer(torch.relu(states))
        return out, states




