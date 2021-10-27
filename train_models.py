"""
Train models for experiment 3.
"""
import torch.nn as nn
from Connectivity import *
from Trials import *
from RNN import *
from skorch import scoring
import random as rdm
import string
from datajoint_tables import *
import itertools
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Get environmental variable 'task_id'
task_id = int(os.environ['SGE_TASK_ID'])
#task_id=0
lr = [.01]
#lambda_r = [0, .01, .1, 1]
lambda_r = [0.01]
lambda_o = [1]
patience = [25]
threshold = [.0001]
batch_size = [180]
param_grid = np.repeat(np.array([x for x in itertools.product( lambda_r, lr, patience, threshold, batch_size,lambda_o)]),repeats=25, axis=0)

parameters = {'lambda_r': param_grid[task_id-1][0],
              'lr': param_grid[task_id-1][1],
              'patience': param_grid[task_id-1][2],
              'threshold': param_grid[task_id-1][3],
              'batch_size': param_grid[task_id-1][4],
              'lambda_o': param_grid[task_id-1][5]}

# Define trial structure.
t = 3000
dt = .2 * 200
n_t = int(round(t / dt))
trial_events = {'n_t': int(round(n_t)),
                'cue_on': int(round(n_t * .1)),
                'cue_off': int(round(n_t * .33)),
                'stim_on': int(round(n_t * .4)),
                'stim_off': int(round(n_t)),
                'dec_on': int(round(n_t * .75)),
                'dec_off': int(round(n_t))}

# Load inputs and labels
inputs, labels, mask, conditions = generate_trials(**trial_events,
                                          n_trials=25,
                                          alpha=0.2,
                                          tau=200,
                                          sigma_in=.01,
                                          baseline=0.2,
                                          n_coh=6)

rnn_net = RNNNet(
    module=RNNModule,
    module__n=100,
    module__connectivity='large',
    module__embedding=False,
    module__radius=1.5,
    module__lambda_r=parameters['lambda_r'],
    module__lambda_o=parameters['lambda_o'],
    module__lambda_w=0,
    module__activation='relu',
    warm_start=False,
    lr=parameters['lr'],
    baseline=.01,
    max_epochs=10000,
    module__mask = mask,
    optimizer=torch.optim.Adam,
    device=device,
    callbacks=[EpochScoring(r2_scorer, on_train=False),
                EpochScoring(L2_rate, on_train=False),
               EpochScoring(L2_weight, on_train=False),
               EpochScoring(L2_ortho, on_train=False),
               EpochScoring(L2_task, on_train=False),
               EarlyStopping(monitor="r2_scorer",
                             patience=parameters['patience'],
                             threshold=parameters['threshold'],
                             lower_is_better=False)])



# Fit rnn
rnn_net.fit(inputs.to(device=device), labels.to(device=device))

# Populate model table
results = {'model_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
           'connectivity': rnn_net.module_.connectivity,
           'n': rnn_net.module_.n,
           'activation':rnn_net.module_.activation,
           'lr': rnn_net.lr,
           'batch_size': rnn_net.batch_size,
           'patience': parameters['patience'],
           'threshold': parameters['threshold'],
           'lambda_r': rnn_net.module_.lambda_r,
           'lambda_o': rnn_net.module_.lambda_o,
           'lambda_w': rnn_net.module_.lambda_w,
           'w_rec': rnn_net.module_.recurrent_layer.weight.data.detach().cpu().numpy(),
           'w_in': rnn_net.module_.input_layer.weight.data.detach().cpu().numpy(),
           'w_out': rnn_net.module_.output_layer.weight.data.detach().cpu().numpy(),
           'embedding': rnn_net.module_.embedding.data.detach().cpu().numpy(),
            'r2': rnn_net.history[-1, 'r2_scorer'].detach().cpu().numpy(),
           'valid_loss': rnn_net.history[-1, 'valid_loss'],
            'train_loss': rnn_net.history[-1, 'train_loss'],
            'epochs': rnn_net.history[-1]['epoch'],
           'l2_ortho': rnn_net.history[-1, 'L2_ortho'].detach().cpu().numpy(),
           'l2_rate': rnn_net.history[-1, 'L2_rate'].detach().cpu().numpy(),
           'l2_weight': rnn_net.history[-1, 'L2_weight'].detach().cpu().numpy(),
           'l2_task': rnn_net.history[-1, 'L2_task'].detach().cpu().numpy(),
           'train_loss_history': np.array(rnn_net.history[:,'train_loss']),
           'valid_loss_history': np.array(rnn_net.history[:,'valid_loss'])}

#if rnn_net.history[-1, 'r2_scorer'].detach().cpu().numpy()>.90:
Model().insert1(results)


