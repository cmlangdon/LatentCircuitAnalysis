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
#task_id = int(os.environ['SGE_TASK_ID'])
task_id = 0
model_ids = (Model_paper()).fetch('model_id')

# Define hyperparameter grid
lr = [.02]
patience = [50]
threshold = [.0001]
batch_size = [128]
sigma_rec = [0.15]
weight_decay=[0]
n_repeats = 25
param_grid = np.repeat(np.array([x for x in itertools.product(model_ids,sigma_rec, lr, patience, threshold, batch_size, weight_decay)]), repeats=n_repeats, axis=0)

# Select hyperparameters for this task
parameters = {'model_id': param_grid[task_id-1][0],
               'sigma_rec': (param_grid[task_id-1][1]).astype(float),
              'lr': (param_grid[task_id-1][2]).astype(float),
              'patience': (param_grid[task_id-1][3]).astype(float),
              'threshold': (param_grid[task_id-1][4]).astype(float),
              'batch_size': (param_grid[task_id-1][5]).astype(int),
              'weight_decay': (param_grid[task_id-1][6]).astype(float)}

# Generate inputs and labels
n_trials = 25
inputs, labels, mask, conditions  = generate_trials(
                                            n_trials=n_trials,
                                            alpha=float(0.2),
                                            tau=200,
                                            sigma_in=.01,
                                            baseline=0.2,
                                            n_coh=6)

# Reconstruct model from model_id
#def simulate_model(model_id):
model_id = parameters['model_id']
N = (Model_paper() & {'model_id':model_id}).fetch1('n')
rnn_net = RNNNet(
    module=RNNModule,
    module__n=N,
    module__connectivity='large',
    module__mask = mask,
    device=device,
)
rnn_net.initialize()
rnn_net.module_.recurrent_layer.weight.data = torch.tensor((Model_paper() & {'model_id':model_id}).fetch1('w_rec'), device= device)
rnn_net.module_.input_layer.weight.data = torch.tensor((Model_paper() & {'model_id':model_id}).fetch1('w_in'),device=device)
rnn_net.module_.output_layer.weight.data = torch.tensor((Model_paper() & {'model_id':model_id}).fetch1('w_out'),device=device)

# Get average trajectories for model
x,_,z = rnn_net.forward(inputs.to(device=device),training=False)
x = x.detach().cpu()
z = z.detach().cpu()
labels = torch.cat((x,z), dim=2)
#     df = pd.DataFrame(data=conditions)
#     df['labels']=list(labels.numpy())
#     df['inputs']=list(inputs.numpy())
#     return df,N
#
#
# def group_mean(x):
#     return np.mean(np.stack(x),axis=0)
#
# df, N = simulate_model(parameters['model_id'])
#
# mean_labels = torch.tensor(np.stack(df.groupby(['context','motion_coh','color_coh'])['labels'].apply(lambda x: group_mean(x)).reset_index()['labels'].values))
# mean_inputs = torch.tensor(np.stack(df.groupby(['context','motion_coh','color_coh'])['inputs'].apply(lambda x: group_mean(x)).reset_index()['inputs'].values))

# Initialize latent nets
input_mask = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0)
output_mask = torch.cat((torch.zeros(2,6),torch.eye(2)),dim=1)
latent_net = LatentNet(
    module=LatentModule,
    module__n=8,
    module__N=N,
    module__alpha = 0.2,
    module__sigma_rec = parameters['sigma_rec'],
    module__weight_decay=parameters['weight_decay'],
    module__input_mask=input_mask.to(device=device),
    module__output_mask=output_mask.to(device=device),
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
               EarlyStopping(monitor="train_loss",
                             patience=parameters['patience'],
                             threshold=parameters['threshold'],
                             lower_is_better=True)]
)
print('Fitting...')

latent_net.fit(inputs.to(device=device), labels.to(device=device))

# Populate LCA table
results = {'model_id': parameters['model_id'],
           'lca_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
           **parameters,
           'alpha': latent_net.module_.alpha.cpu().numpy(),
           'sigma_rec': latent_net.module_.sigma_rec.cpu().numpy(),
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
LCA_paper.insert1(results)

