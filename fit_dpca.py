"""
Fit dpca for a model and return projection
"""


from datajoint_tables import *
import numpy as np
from dPCA import dPCA
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns


def fit_dpca(model_id):

    n = (Model() & {'model_id': model_id}).fetch1('n')

    contexts = ['motion', 'color']
    motion_coherences = np.linspace(-.2, .2, 6)
    color_coherences = np.linspace(-.2, .2, 6)

    X = np.zeros((n, 2, 6, 6, 75))
    for i in range(2):
        for j in range(6):
            for k in range(6):
                X[:, i, j, k, :] = np.mean(np.stack((Trial() & {'model_id': model_id,'context': contexts[i], 'motion_coh': np.around(motion_coherences[j], 2),
                          'color_coh': np.around(color_coherences[k], 2)}).fetch('hidden')), axis=0).T

    query = Trial() & {'model_id': model_id}
    Xtrial = np.zeros((50, n, 2, 6, 6, 75))
    for i in range(2):
        for j in range(6):
            for k in range(6):
                Xtrial[:, :, i, j, k, :] = np.transpose(np.stack((query & {'context': contexts[i], 'motion_coh': np.around(motion_coherences[j],2), 'color_coh': np.around(color_coherences[k],2)}).fetch('hidden')),(0,2,1))

    dpca = dPCA.dPCA(labels='rmct', regularizer='auto')
    dpca.protect = ['t']
    
    q = np.concatenate((dpca.D['r'][:, 0, None], dpca.D['m'][:, 0, None], dpca.D['c'][:, 0, None], dpca.D['m'][:, 0, None]),axis=1)

    return q