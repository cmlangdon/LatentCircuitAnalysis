"""
Train models for experiment 3.
"""
import torch.nn as nn
from Connectivity import *
from Trials import *
from RNN import *
from psychometrics import *
from skorch import scoring
import random as rdm
import string
from datajoint_tables import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
                                          n_trials=50,
                                          alpha=0.2,
                                          tau=200,
                                          sigma_in=.01,
                                          baseline=0.2,
                                          n_coh=6)

rnn_net = RNNNet(
    module=RNNModule,
    module__n=150,
    module__connectivity='large',
    module__radius=1.5,
    module__lambda_r=0.05,
    module__lambda_o=0,
    module__lambda_w=0,
    module__lambda_i=0,
    warm_start=False,
    lr=.001,
    baseline=.01,
    max_epochs=2000,
    batch_size=180,
    module__mask = mask,
    optimizer=torch.optim.Adam,
    device=device,
    callbacks=[r2,
               EpochScoring(make_scorer(L2_rate), on_train=False),
               EpochScoring(L2_weight, on_train=False),
               EpochScoring(L2_ortho, on_train=False),
               EpochScoring(L2_invar, on_train=False),
               EpochScoring(make_scorer(L2_task), on_train=False),
               EpochScoring(stopping, on_train=False),
               EarlyStopping(monitor="stopping",
                             patience=10,
                             threshold=0.01,
                             threshold_mode='abs',
                             lower_is_better=False)]

)

# Fit rnn
rnn_net.fit(inputs.to(device=device), labels.to(device=device))

# Populate model table
results = {'model_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
           'connectivity': rnn_net.module_.connectivity,
           'n': rnn_net.module_.n,
           'lr': rnn_net.lr,
           'batch_size': rnn_net.batch_size,
           'lambda_r': rnn_net.module_.lambda_r,
           'lambda_i': rnn_net.module_.lambda_i,
           'lambda_o': rnn_net.module_.lambda_o,
           'lambda_w': rnn_net.module_.lambda_w,
           'w_rec': rnn_net.module_.recurrent_layer.weight.data.detach().cpu().numpy(),
           'w_in': rnn_net.module_.input_layer.weight.data.detach().cpu().numpy(),
           'w_out': rnn_net.module_.output_layer.weight.data.detach().cpu().numpy(),
            'r2': rnn_net.history[-1, 'r2_scorer'].detach().cpu().numpy(),
           'valid_loss': rnn_net.history[-1, 'valid_loss'],
           'l2_invar': rnn_net.history[-1, 'L2_invar'],
           'l2_ortho': rnn_net.history[-1, 'L2_ortho'].detach().cpu().numpy(),
           'l2_rate': rnn_net.history[-1, 'L2_rate'].detach().cpu().numpy(),
           'l2_weight': rnn_net.history[-1, 'L2_weight'].detach().cpu().numpy(),
           'l2_task': rnn_net.history[-1, 'L2_task'].detach().cpu().numpy(),
           'train_loss_history': np.array(rnn_net.history[:,'train_loss']),
           'valid_loss_history': np.array(rnn_net.history[:,'valid_loss'])}
Model().insert1(results)

# Populate trial table
n_trials = len(conditions)

z_masked, z, y = rnn_net.forward(inputs, training=False)
z = z.detach().cpu().numpy()
y = y.detach().cpu().numpy()
for k in range(n_trials):
    Trial().insert1({'model_id': results['model_id'],
                     'trial_id': k,
                     **conditions[k],
                     'input': inputs[k].numpy(),
                     'hidden': y[k,:,:],
                     'output': z[k],
                     'score': r2_scorer(labels[None,k,:,:], z_masked[None,k,:,:]).detach().cpu().numpy().item()
                       })
