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
       lr: decimal(6,6)                       # learning rate
       batch_size: int
       patience: int
       threshold: Decimal(6,5)
       lambda_r: decimal(8,6)                 # firing rate regularization constant
       lambda_o: decimal(8,6)                 # orthogonality regularization constant
       lambda_w: decimal(8,6)                 # weight regularization
       r2: float
       valid_loss: float
       train_loss: float
       epochs: int
       l2_ortho: float
       l2_rate: float
       l2_weight: float
       l2_task: float
        w_rec: longblob
       w_in: longblob
       w_out: longblob
       embedding=NULL: longblob
       train_loss_history: longblob
       valid_loss_history: longblob
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
class LCA(dj.Manual):
    definition = """
    -> Model
    lca_id: char(8)                   
    ---
    alpha: Decimal(3,2)
    sigma_rec: Decimal(3,2)
    lr: Decimal(8,7)
    weight_decay: Decimal(6,5)
    patience: int
    threshold: Decimal(8,7)
    n_trials: int
    batch_size: int
    max_epochs: int
    epochs: int
    r2_x: Decimal(5,4)
    r2_xqt: Decimal(5,4)
    r2_z: Decimal(5,4)
    valid_loss: Decimal(6,5)
    train_loss: Decimal(6,5)
    valid_loss_history: longblob
    train_loss_history: longblob
    w_rec: longblob
    w_in: longblob
    w_out: longblob
    q: longblob
    a: longblob
    w_error=NULL: float
    q_error=NULL: float
    """

#Changed loss function to MSE(x,xbar@q)/varx + MSE(z,zbar)/varz
@schema
class LCA_unconstrained(dj.Manual):
    definition = """
    -> LCA
    lca2_id: char(8)                   
    ---
    alpha: Decimal(3,2)
    sigma_rec: Decimal(3,2)
    lr: Decimal(8,7)
    weight_decay: Decimal(6,5)
    patience: int
    threshold: Decimal(8,7)
    n_trials: int
    batch_size: int
    max_epochs: int
    epochs: int
    r2_x: Decimal(5,4)
    r2_xqt: Decimal(5,4)
    r2_z: Decimal(5,4)
    valid_loss: Decimal(6,5)
    train_loss: Decimal(6,5)
    valid_loss_history: longblob
    train_loss_history: longblob
    w_rec: longblob
    w_in: longblob
    w_out: longblob
    q: longblob
    a: longblob
    w_error=NULL: float
    q_error=NULL: float
    """




@schema
class ModelPerturbation(dj.Manual):
    definition = """
    -> Model   
    perturbation_id: char(8)    
    ---
    type: enum('lca', 'tdr', 'lr', 'dpca')
    direction: enum('context','motion','color','choice')        
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

@schema
class Regression(dj.Manual):
    definition = """
    -> Model
    orthogonal: enum('True', 'False')
    active: enum('True', 'False')               
    ---
    q: longblob
    """