"""
Given a model, fit a latent circuit and store results in LCA table.
"""

import pandas as pd
import os
from datajoint_tables import *
import string
import random as rdm
from LatentCircuitAnalysis import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Device: ' + device)
# task_id = int(os.environ['SGE_TASK_ID'])

# Load data for model
model_id = "fdtFi2Nw"
print('Loading model dimension...')
#N = (Model() & {'model_id': model_id}).fetch1('n')
N=150
print('Loading data...')
#inputs, labels, conditions = load_model_data(model_id)
inputs, labels, conditions = load_embedded_model_data(model_id)

# Choose hyperparameters
hyperparameters = {'lr': .001,
                   'max_epochs': 3000}

# Initialize latent nets
recurrent_mask = torch.ones(8, 8).float().to(device=device)
recurrent_mask = recurrent_mask - torch.diag(torch.diag(recurrent_mask))
input_mask = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0)
output_mask = torch.cat((torch.zeros(2,6),torch.eye(2)),dim=1)
latent_net = LatentNet(
    module=LatentModule,
    module__n=8,
    module__N=N,
    module__recurrent_mask=recurrent_mask.to(device=device),
    module__input_mask=input_mask.to(device=device),
    module__output_mask=output_mask.to(device=device),
    warm_start=False,
    lr=hyperparameters['lr'],
    max_epochs=hyperparameters['max_epochs'],
    optimizer=torch.optim.Adam,
    device=device
)
print('Fitting...')
# Fit LCA
latent_net.fit(inputs, labels)

# Compute w_error and q_error:
model_A = torch.zeros((8,14))
model_A[:8,:8] = torch.tensor((Model() & {'model_id':  model_id}).fetch1('w_rec'))
model_A[:8,8:] = torch.tensor((Model() & {'model_id':  model_id}).fetch1('w_in'))

latent_A = torch.zeros((8,14))
latent_A[:8,:6] = latent_net.module_.recurrent_layer.weight.data
latent_A[:8,8:] = latent_net.module_.input_layer.weight.data

w_error = torch.norm(model_A-latent_A) / torch.norm(model_A)
q_true = (EmbeddedTrial() & {'model_id':model_id})
q_error = torch.norm(latent_net.module_.q.weight.data - q_true) / torch.norm(q_true)

# Populate LCA table
results = {'model_id': model_id,
           'lca_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
           **hyperparameters,
           'valid_loss': latent_net.history[-1, 'valid_loss'],
           'valid_loss_history': np.array(latent_net.history[:, 'valid_loss']),
           'train_loss_history': np.array(latent_net.history[:, 'train_loss']),
           'w_rec': latent_net.module_.recurrent_layer.weight.data.detach().cpu().numpy(),
           'w_in': latent_net.module_.input_layer.weight.data.detach().cpu().numpy(),
            'w_out': latent_net.module_.output_layer.weight.data.detach().cpu().numpy(),
           'q': latent_net.module_.q.detach().cpu().numpy(),
            'w_error': w_error.detach().cpu().numpy(),
            'q_error': q_error.detach().cpu().numpy()}
LCA.insert1(results)

# Populate LCATrial table
x_pred, z_pred = latent_net.forward(inputs)
#y_pred = y_pred.reshape(y_pred.shape[0], -1, latent_net.module_.N)

y_pred = np.concatenate((x_pred, z_pred), axis=2)
print(y_pred.shape)
for k in range(y_pred.shape[0]):
    LCATrial().insert1({'model_id': model_id,
                        'lca_id': results['lca_id'],
                        'trial_id': k,
                        **conditions[k],
                        'y_pred': y_pred[k, :, :]})
