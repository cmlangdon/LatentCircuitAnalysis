"""
Implementation of Targeted Dimensionality Reduction (TDR)
"""
from datajoint_tables import *
from sklearn.decomposition import PCA
import pandas as pd
from RNN import *
from Trials import *


def tdr(model_id):
    
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

    z_mask, z, r = rnn.forward(inputs)
    r = r.detach().numpy()
    #labels = torch.reshape(x,(-1,x.shape[2])) 

    conditions = pd.DataFrame(conditions).values
    conditions[conditions[:,0]=='motion',0]=1
    conditions[conditions[:,0]=='color',0]=-1
    conditions[:,1]= 5 * conditions[:,1]
    conditions[:,2]= 5 * conditions[:,2]
    
    
    
    

    # Get trial by trial responses, x
    #trial_query = Trial() & {'model_id': model_id}
    #r = np.stack(trial_query.fetch('hidden'), 0)
    n = r.shape[2]
    #active_neurons = np.argwhere(np.std(r, axis=(0,1)) > .1)[:, 0]

    #r = r[:,:,active_neurons]
    # Standardize neuron responses
    means = np.mean(r, axis=(0, 1))
    stds = np.std(r, axis=(0, 1))
    r = (r-means) / stds

    # Create matrix F of trial conditions.
    #trial_conditions = ((Trial() & {'model_id': model_id}).proj('context', 'motion_coh','color_coh','correct_choice')).fetch()
#     n_trials = len(trial_conditions)
#     F = []
#     for i in range(n_trials):
#         f = []
#         f.append(trial_conditions[i][5])
#         f.append(5 * float(trial_conditions[i][3]))
#         f.append(5 * float(trial_conditions[i][4]))
#         if trial_conditions[i][2] == 'motion':
#             f.append(1)
#         else:
#             f.append(-1)
#         #f.append(1)
#         F.append(f)

#     F = np.array(np.vstack(F)).T
    
    F= conditions.astype(float).T
    print(F.shape)
    # Estimate regression coefficients
    n_neurons = r.shape[2]
    n_time = r.shape[1]
    n_coeff = 4
    B = np.zeros((n_coeff, n_time, n_neurons))
    for i in range(n_neurons):
        for j in range(n_time):
            B[:, j, i] = np.linalg.pinv(F @ F.T) @ F @ r[:, j, i]


    B_max = np.zeros((n_coeff, n_neurons))
    for i in range(n_coeff):
        t_v_max = np.argmax(np.linalg.norm(B[i,:,:],axis=1))
        B_max[i, :] = B[i, t_v_max, :]



    # Orthogonalize regression vectors
    Q, R = np.linalg.qr(B_max.T)
    #b = np.zeros((4,n))
    #b[:, active_neurons] = B_max
    #b = b[[3,1,2,0],:]
    #B_max = B_max[[3,1,2,0],:]
    #Q = Q[:,[3,1,2,0]].T
    return B_max, Q