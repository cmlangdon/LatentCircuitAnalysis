"""
For a given model, embed trials using a random orthogonal q.
"""
import torch
from datajoint_tables import *

model_id = "fdtFi2Nw"
n = (Model() & {'model_id': model_id}).fetch1('n')
N = 150

# Construct random orthogonal Q
A = np.random.rand(N, N)
Q = (np.eye(N) - (A - A.T) / 2) @ np.linalg.inv(np.eye(N) + (A - A.T) / 2)
q = Q[:n, :]

trial_data = (Trial() & {'model_id': model_id}).fetch(as_dict=True)

for k in range(len(trial_data)):
    print(k)
    EmbeddedTrial().insert1({'model_id': model_id,
                     'trial_id': k,
                     'context':trial_data[k]['context'],
                    'motion_coh': trial_data[k]['motion_coh'],
                    'color_coh': trial_data[k]['color_coh'],
                    'correct_choice': trial_data[k]['correct_choice'],
                     'input': trial_data[k]['input'],
                     'hidden': trial_data[k]['hidden'] @ q,
                     'output': trial_data[k]['output'],
                    'q': q
                       })
