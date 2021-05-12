import datajoint as dj
import torch
import numpy as np
import skorch
from skorch.helper import SliceDict

dj.config['database.host'] = 'pioneer.cshl.edu'
dj.config['database.user'] = 'langdon'
dj.config['database.password'] = 'bHaaratc'
dj.config['display.limit'] = 100
dj.config["enable_python_native_blobs"] = True
dj.conn(reset=True)
schema = dj.schema('langdon_rnn')


@schema
class Model(dj.Manual):
    definition = """
       # model table
       model_id: char(8)                      # unique model id
       ---
       connectivity: enum('small', 'large')   # specify connectivity structure
       n: int                                 # number of neurons

       tau: decimal(3,0)                      # time constant
       sigma_rec: decimal(4,4)                # recurrent noise
       radius: decimal(3,2)                   # spectral radius used for initialization of w_rec
       alpha: decimal(2,2)                    # dt/tau
       lr: decimal(6,6)                       # learning rate
       batch_size: int
       loss_target: float                     # training stops when loss reaches this goal
       lambda_r: decimal(8,6)                 # firing rate regularization constant
       lambda_w: decimal(8,6)                 # weight regularization constant
       lambda_o: decimal(8,6)                 # orthogonality regularization constant
       w_rec: longblob
       w_in: longblob
       w_out: longblob
       loss_curve: longblob
       """



@schema
class Trial(dj.Manual):
    definition = """
    -> Model
    trial_id: int                   
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    correct_choice: int
    input: longblob
    hidden: longblob
    output: longblob
    choice: int
    mse: decimal(6,4)
    """



@schema
class ModelPerturbation(dj.Manual):
    definition = """
    -> Model   
    perturbation_id: char(8)    
    ---
    direction: int        
    strength: decimal(5,3)
    """


@schema
class PerturbationTrial(dj.Manual):
    definition = """
    -> ModelPerturbation   
    trial_id: int                   
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    input: longblob
    hidden: longblob
    output: longblob
    choice: int
    mse: decimal(6,4)
    """


@schema
class AveragePerturbationTrial(dj.Manual):
    definition = """
    -> ModelPerturbation
    condition_id: int                   
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    input: longblob
    hidden: longblob
    output: longblob
    mse: decimal(6,4)
    prob_right: decimal(3,2)
    """


def load_model_data(model_id):
    trial_query = Trial() & {'model_id': model_id}
    u = torch.tensor(np.stack(trial_query.fetch('input'), 0)).float()
    x = torch.tensor(np.stack(trial_query.fetch('hidden'), 0)).float()
    z = torch.tensor(np.stack(trial_query.fetch('output'), 0)).float()
    #y0 = y[:, None, 0, :]
    inputs = u
    labels = torch.cat((x,z),dim=2)
    conditions = trial_query.fetch('context', 'motion_coh', 'color_coh', as_dict=True)
    return inputs, labels, conditions

def load_embedded_model_data(model_id):
    trial_query = EmbeddedTrial() & {'model_id': model_id}
    u = torch.tensor(np.stack(trial_query.fetch('input'), 0)).float()
    x = torch.tensor(np.stack(trial_query.fetch('hidden'), 0)).float()
    z = torch.tensor(np.stack(trial_query.fetch('output'), 0)).float()
    #y0 = y[:, None, 0, :]
    inputs = u
    labels = torch.cat((x,z),dim=2)
    conditions = trial_query.fetch('context', 'motion_coh', 'color_coh', as_dict=True)
    return inputs, labels, conditions


@schema
class LCA(dj.Manual):
    definition = """
    -> Model
    lca_id: char(8)                   
    ---
    lr: Decimal(8,7)
    max_epochs: int
    valid_loss: Decimal(5,4)
    valid_loss_history: longblob
    train_loss_history: longblob
    w_rec: longblob
    w_in: longblob
    w_out: longblob
    q: longblob
    w_error=NULL: float
    q_error=NULL: float
    """


@schema
class LCATrial(dj.Manual):
    definition = """
    -> LCA   
    trial_id: int        
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    y_pred: longblob
    """


@schema
class ModelPerturbation(dj.Manual):
    definition = """
    -> Model   
    perturbation_id: char(8)    
    ---
    direction: int        
    strength: decimal(5,3)
    """

@schema
class PerturbationTrial(dj.Manual):
    definition = """
    -> ModelPerturbation   
    trial_id: int                   
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    correct_choice: int
    input: longblob
    hidden: longblob
    output: longblob
    """


@schema
class EmbeddedTrial(dj.Manual):
    definition = """
    -> Trial
    trial_id: int                   
    ---
    context: enum("motion", "color")
    motion_coh: decimal(2,2)
    color_coh: decimal(2,2)
    correct_choice: int
    input: longblob
    hidden: longblob
    output: longblob
    q: longblob
    """