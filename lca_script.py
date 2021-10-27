"""
Given a model, fit a latent circuit and store results in LCA table.
"""

import pandas as pd
import os
from datajoint_tables import *
import string
import random as rdm
from LatentCircuitAnalysis import *
from RNN import *
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Device: ' + device)

# Get environmental variable 'task_id'
task_id = int(os.environ['SGE_TASK_ID'])
#task_id=0
# Get model ids

model_ids = (Model() & 'n=100').fetch('model_id')
lr = [.02]
#lr= list(np.round(10**np.linspace(-3,-1,25),4))
patience = [50]
threshold = [.0001]
batch_size = [128]
sigma_rec = [0.15]
weight_decay=[0]
param_grid = np.repeat(np.array([x for x in itertools.product(model_ids,sigma_rec, lr, patience, threshold, batch_size, weight_decay)]), repeats=5, axis=0)

parameters = {'model_id': param_grid[task_id-1][0],
               'sigma_rec': (param_grid[task_id-1][1]).astype(float),
              'lr': (param_grid[task_id-1][2]).astype(float),
              'patience': (param_grid[task_id-1][3]).astype(float),
              'threshold': (param_grid[task_id-1][4]).astype(float),
              'batch_size': (param_grid[task_id-1][5]).astype(int),
              'weight_decay': (param_grid[task_id-1][6]).astype(float)}


# Load data for model
model_id = parameters['model_id']
print(model_id)
N = (Model() & {'model_id': model_id}).fetch1('n')

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
n_trials = 64
inputs, _, mask, conditions = generate_trials(**trial_events,
                                          n_trials=n_trials,
                                          alpha=0.2,
                                          tau=200,
                                          sigma_in=.01,
                                          baseline=0.2,
                                          n_coh=6)

n = (Model() & {'model_id':model_id}).fetch1('n')
rnn_net = RNNNet(
    module=RNNModule,
    module__n=n,
    module__connectivity='large',
    baseline=.015,
    module__mask = mask,
    device=device,
)
rnn_net.initialize()

rnn_net.module_.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'), device= device)
rnn_net.module_.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'),device=device)
rnn_net.module_.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'),device=device)

x,_,z = rnn_net.forward(inputs.to(device=device),training=False)
q_true = torch.tensor((Model() & {'model_id': model_id}).fetch1('embedding')).float()
x = x.detach().cpu() @ q_true
z = z.detach().cpu()
labels = torch.cat((x,z), dim=2)

# Initialize latent nets
recurrent_mask = torch.ones(8, 8).float().to(device=device)
recurrent_mask = recurrent_mask - torch.diag(torch.diag(recurrent_mask))
input_mask = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0)
output_mask = torch.cat((torch.zeros(2,6),torch.eye(2)),dim=1)
latent_net = LatentNet(
    module=LatentModule,
    module__n=8,
    module__N=N,
    module__alpha = 0.2,
    module__sigma_rec = parameters['sigma_rec'],
    module__weight_decay=parameters['weight_decay'],
    module__recurrent_mask=recurrent_mask.to(device=device),
    module__input_mask=input_mask.to(device=device),
    module__output_mask=output_mask.to(device=device),
    module__activation='relu',
    warm_start=False,
    lr=parameters['lr'],
    batch_size=int(parameters['batch_size']),
    max_epochs=10000,
    optimizer=torch.optim.Adam,
    device=device,
    callbacks=[EpochScoring(r2_x, on_train=False),
               EpochScoring(r2_xqt, on_train=False),
                EpochScoring(r2_z, on_train=False),
                EpochScoring(rsquared, on_train=False),
               EarlyStopping(monitor="valid_loss",
                             patience=parameters['patience'],
                             threshold=parameters['threshold'],
                             lower_is_better=True)]
)
print('Fitting...')
# Fit LCA
latent_net.fit(inputs, labels)

# Compute w_error and q_error:
# w_rec = (Model() & {'model_id': model_id}).fetch1('w_rec')
# w_rec_true = latent_net.module_.recurrent_layer.weight.data.detach().cpu().numpy()
# w_error = np.linalg.norm(w_rec-w_rec_true) / np.linalg.norm(w_rec)
# q_error = np.linalg.norm(latent_net.module_.q.detach().cpu().numpy() - q_true.numpy()) / np.linalg.norm(q_true.numpy())

# Populate LCA table
results = {'model_id': parameters['model_id'],
           'lca_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
           **parameters,
           'alpha': latent_net.module_.alpha.cpu().numpy(),
           'sigma_rec': latent_net.module_.sigma_rec.cpu().numpy(),
           'activation': latent_net.module_.activation,
           'weight_decay': parameters['weight_decay'],
           'n_trials': n_trials,
           'batch_size': latent_net.batch_size,
            'r2_x':latent_net.history[-1, 'r2_x'].detach().cpu().numpy(),
           'r2_xqt':latent_net.history[-1, 'r2_xqt'].detach().cpu().numpy(),
            'r2_z':latent_net.history[-1, 'r2_z'].detach().cpu().numpy(),
            'r2':latent_net.history[-1, 'rsquared'].detach().cpu().numpy(),
           'valid_loss': latent_net.history[-1, 'valid_loss'],
            'train_loss': latent_net.history[-1, 'train_loss'],
           'valid_loss_history': np.array(latent_net.history[:, 'valid_loss']),
           'train_loss_history': np.array(latent_net.history[:, 'train_loss']),
           'epochs': latent_net.history[-1]['epoch'],
           'max_epochs': latent_net.max_epochs,
           'w_rec': latent_net.module_.recurrent_layer.weight.data.detach().cpu().numpy(),
           'w_in': latent_net.module_.input_layer.weight.data.detach().cpu().numpy(),
            'w_out': latent_net.module_.output_layer.weight.data.detach().cpu().numpy(),
           'q': latent_net.module_.q.detach().cpu().numpy(),
           'a': latent_net.module_.A.detach().cpu().numpy()}
            #'w_error': w_error,
            #'q_error': q_error}
LCA.insert1(results)

