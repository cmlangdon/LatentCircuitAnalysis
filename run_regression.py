from datajoint_tables import *
from Regressor import *
from skorch.callbacks import EpochScoring
from skorch.callbacks import EarlyStopping
from Trials import *
from RNN import *
import pandas as pd

model_id = "5a8Myb4M"
# trial_query = Trial() & {'model_id': model_id}
# conditions = np.stack(trial_query.fetch('context', 'motion_coh', 'color_coh','correct_choice')).T
# conditions[conditions[:,0]=='motion',0]=1
# conditions[conditions[:,0]=='color',0]=-1
# conditions[:,1]= 5 * conditions[:,1]
# conditions[:,2]= 5 * conditions[:,2]
# conditions = torch.tensor(conditions.astype(float)).float()
# inputs = np.repeat(conditions, repeats=75, axis=0)
# x = torch.tensor(np.stack(trial_query.fetch('hidden'), 0)).float()
# labels = torch.reshape(x,(-1,x.shape[2]))


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
                                          n_trials=15,
                                          alpha=0.2,
                                          tau=200,
                                          sigma_in=.01,
                                          baseline=0.2,
                                          n_coh=6)


n = (Model() & {'model_id':model_id}).fetch1('n')
rnn = RNNModule(connectivity="large", mask=mask, n=n)

rnn.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'))
rnn.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'))
rnn.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'))

z_mask, z, x = rnn.forward(inputs)
x = x.detach()
labels = torch.reshape(x,(-1,x.shape[2])) 

conditions = np.repeat(pd.DataFrame(conditions).values, repeats=75, axis=0)
conditions[conditions[:,0]=='motion',0]=1
conditions[conditions[:,0]=='color',0]=-1
conditions[:,1]= 5 * conditions[:,1]
conditions[:,2]= 5 * conditions[:,2]
inputs = torch.tensor(conditions.astype(float)).float()

from skorch import scoring
orthogonal = True
active = True

if active:
    active_neurons = np.argwhere(torch.std(labels,dim=0) > .1)[0]
    labels = labels[:, active_neurons]


regression_net = RegressionNet(
    module=RegressionModule,
    module__orthogonal= orthogonal,
    module__N = labels.shape[1],
    warm_start=False,
    lr=.001,
    max_epochs=50,
    optimizer=torch.optim.Adam,
    device=device,
    callbacks=[EarlyStopping(monitor='valid_loss',
                             patience=10,
                             threshold=0.001,
                             threshold_mode='abs',
                             lower_is_better=True)]
)


regression_net.fit(inputs,labels)

if active:
    v = torch.zeros((4,x.shape[2]))
    v[:,active_neurons] = regression_net.module_.q
    regression_net.module_.q = v

# Populate Regression table
results = {'model_id':model_id,
           'orthogonal': str(orthogonal),
           'active': str(active),
           'q': regression_net.module_.q.detach().cpu().numpy()}

Regression.insert1(results)

