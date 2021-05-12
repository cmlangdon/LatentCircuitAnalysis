"""
For a given model and perturbation, simulate perturbation and populate PerturbationTrial and AveragePerturbationTrial
tables.
"""
from RNN import *
from datajoint_tables import *
from Trials import *
import random as rdm
import string
import torch
from psychometrics import *
import pandas as pd

if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'
    print('cpu')


# Define perturbation class
class Perturbation(nn.Module, ABC):

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


# Initialize perturbation class with connectivity from model.
model_id = "6rV5JuQH"
query = Model() & {'model_id': model_id}
perturbation = Perturbation(tau=200,
                            sigma_rec=float(0.15),
                            n=query.fetch1('n'))
perturbation.input_layer.weight.data = torch.tensor(query.fetch1('w_in')).float()
perturbation.recurrent_layer.weight.data = torch.tensor(query.fetch1('w_rec')).float()
perturbation.output_layer.weight.data = torch.tensor(query.fetch1('w_out')).float()
perturbation.to(device=device)

# Define trial structure.
t = 3000
dt = 200 * .2
n_t = int(round(t / dt))
trial_events = {'n_t': int(round(n_t)),
                'cue_on': int(round(n_t * .1)),
                'cue_off': int(round(n_t * .33)),
                'stim_on': int(round(n_t * .4)),
                'stim_off': int(round(n_t)),
                'dec_on': int(round(n_t * .75)),
                'dec_off': int(round(n_t))}

# Generate dataset for simulation

inputs, labels, mask, conditions  = generate_trials(**trial_events,
                                            n_trials=25,
                                            alpha=float(0.2),
                                            tau=perturbation.tau,
                                            sigma_in=.01,
                                            baseline=0.2,
                                            n_coh=6)
# Define perturbation
lca_id = 'mrhq0Ftn'
lca_query = LCA() & {'lca_id': lca_id}
direction = 2
strength = .1
q = lca_query.fetch('q')[0]
p = strength * q[None, direction, :]
p = torch.tensor(p).float().to(device=device)


inputs = inputs.to(device)
output, hidden = perturbation(inputs, p, float(0.2))


# Populate model perturbation table
results = {'model_id': model_id,
           'perturbation_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=3)),
           'direction': direction,
           'strength': strength}
ModelPerturbation().insert1({**results})

# Populate trial table
n_trials = len(conditions)
for k in range(n_trials):
    print(k)
    PerturbationTrial().insert1({'model_id': model_id,
                                 'perturbation_id': results['perturbation_id'],
                                 'trial_id': k,
                                 **conditions[k],
                                 'input': inputs[k,:,:].detach().cpu().numpy(),
                                 'hidden': hidden[k,:,:].detach().cpu().numpy(),
                                 'output': output[k,:,:].detach().cpu().numpy()})

