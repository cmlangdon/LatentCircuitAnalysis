"""
For a given model, take random Q and embedded the hidden states for each trial.
"""
import torch
import numpy as np
from datajoint_tables import *

N = 150
model_id="6rV5JuQH"
lca_id = "RdzDulrS"
n = 8
model_data = (LCATrial() & {'lca_id': lca_id}).fetch(as_dict=True)

A = torch.nn.Parameter(torch.rand(N, N))
Q = (torch.eye(N) - (A - A.t()) / 2) @ torch.inverse(torch.eye(N) + (A - A.t()) / 2)
q = Q[:n, :].detach().numpy()

for i in range(len(model_data)):
    print(i)
    trial_data = {
        'model_id': model_id,
        'lca_id': lca_id,
                 'trial_id': model_data[i]['trial_id'],
                'context':model_data[i]['context'],
                'motion_coh':model_data[i]['motion_coh'],
                'color_coh':model_data[i]['color_coh'],
                'q': q,
                'xq': model_data[i]['y_pred'][:,:-2] @ q}
    EmbeddedLCATrial().insert1(trial_data)

