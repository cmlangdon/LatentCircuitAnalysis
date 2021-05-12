
import torch
from sklearn.metrics import make_scorer
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from skorch.utils import to_tensor
from torch.nn import functional as F


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class LatentModule(torch.nn.Module):
    def __init__(self, recurrent_mask, input_mask,output_mask, n, N):
        super(LatentModule, self).__init__()

        self.recurrent_mask = recurrent_mask
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.N = N
        self.n = n
        self.input_size = 6

        self.recurrent_layer = torch.nn.Linear(self.n, self.n, bias=False)

        self.input_layer = torch.nn.Linear(6, self.n, bias=False)
        self.input_layer.weight.data = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0).float().to(device=device)
        self.input_layer.weight.requires_grad = True

        self.output_layer = torch.nn.Linear(self.N, 2, bias=False)
        self.output_layer.weight.data = torch.cat((torch.zeros(2,self.n-2),torch.eye(2)),dim=1).float().to(device=device)
        self.output_layer.weight.requires_grad = True

        #self.x0 = torch.nn.Parameter(torch.zeros(1, 1, self.n, device=device), requires_grad=True)

        #self.q = torch.nn.Linear(self.n, self.N, bias=False)
        #self.q.weight.data = self.q.weight.data / torch.norm(self.q.weight.data,dim=0)

        self.A = torch.nn.Parameter(torch.rand(self.N, self.N,device=device), requires_grad=True)
        self.Q = (torch.eye(self.N, device=device) - (self.A - self.A.t()) / 2) @ torch.inverse(torch.eye(self.N,device=device) + (self.A - self.A.t()) / 2)
        self.q = self.Q[:self.n, :]
        self.q = self.q.to(device=device)

    def forward(self, u):
        #u = u.reshape(u.shape[0], -1, self.input_size)
        #y_0 = y_0.reshape(y_0.shape[0], 1, -1)
        t = u.shape[1]
        #x = y_0 @ self.q.weight.data.t()
        #x = self.x0
        x = torch.zeros(u.shape[0],1,self.n).to(device=device)
        for i in range(t - 1):
            #x_new = self.recurrent_layer(x[:, i, :]) + self.input_layer(u[:, i, :])
            #noise = torch.sqrt(torch.tensor(2 * 0.2 * 0.15 ** 2)) * torch.empty(u.shape[0], t, self.n).normal_(mean=0,std=1).to(device=device)
            x_new = (1 - .2) * x[:, i, :] + .2 * (
                torch.relu(self.recurrent_layer(x[:, i, :]) + self.input_layer(u[:, i, :]) ))

            x = torch.cat((x, x_new.unsqueeze_(1)), 1)
        z = torch.relu(self.output_layer(x))
        #y_pred = x @ self.q
        #y_pred = self.q(x)
        return x, z
        #return y_pred.reshape(-1, y_pred.shape[1] * y_pred.shape[2])

    def cayley_transform(self):
        skew = (self.A - self.A.t()) / 2
        skew = skew.to(device=device)
        eye = torch.eye(self.N).to(device=device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        self.q = o[:self.n, :]


def r2_scorer(y_true, y_pred):
    y_true = to_tensor(y_true,device=device)
    y_pred = to_tensor(y_pred,device=device)
    var_y = torch.var(y_true, unbiased=False)

    return 1-F.mse_loss(y_pred, y_true,reduction='mean') / var_y

r2 = EpochScoring(scoring=make_scorer(r2_scorer), on_train=False)


class LatentNet(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        xbar = y_pred[0]
        zbar = y_pred[1]

        x = y_true[:, :, :-2]
        z = y_true[:, :, -2:]
        xqtq = x @ self.module_.q.t() @ self.module_.q
        xqt = x @ self.module_.q.t()

        return self.criterion_(x, xqtq) / self.criterion_(x, torch.zeros_like(x)) + \
               self.criterion_(xqt, xbar) / self.criterion_(xqt, torch.zeros_like(xqt)) +\
               self.criterion_(z, zbar) / self.criterion_(z, torch.zeros_like(z))


    def train_step(self, Xi, yi, **fit_params):
        """Prepares a loss function callable and pass it to the optimizer,
        hence performing one optimization step.

        Loss function callable as required by some optimizers (and accepted by
        all of them):
        https://pytorch.org/docs/master/optim.html#optimizer-step-closure

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the train_split call.

        """
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)
        #self.module_.recurrent_layer.weight.data = self.module_.recurrent_mask * self.module_.recurrent_layer.weight.data # zero diagonal on recurrent
        #self.module_.input_layer.weight.data =  torch.relu(self.module_.input_layer.weight.data)
        #self.module_.input_layer.weight.data = self.module_.input_mask * self.module_.input_layer.weight.data # diagonal input
        #self.module_.q.weight.data = self.module_.q.weight.data / torch.norm(self.module_.q.weight.data, dim=0) # normalized q
        self.module_.input_layer.weight.data = self.module_.input_mask * torch.relu(self.module_.input_layer.weight.data)
        self.module_.output_layer.weight.data = self.module_.output_mask * torch.relu(
            self.module_.output_layer.weight.data)

        #self.module_.input_layer.weight.data = self.module_.input_mask
        self.module_.cayley_transform()
        #norm = self.module_.regression_layer.weight.data.norm(p=2, dim=0, keepdim=True)
        #self.module_.regression_layer.weight.data = self.module_.regression_layer.weight.data.div(norm.expand_as(self.module_.regression_layer.weight.data))
        #self.module_.regression_layer.weight.data = torch.relu(self.module_.regression_layer.weight.data)

        return step_accumulator.get_step()






