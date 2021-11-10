
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
    def __init__(self,  input_mask,output_mask, n, N, alpha, sigma_rec, weight_decay):
        super(LatentModule, self).__init__()

        self.input_mask = input_mask
        self.output_mask = output_mask
        self.N = N
        self.n = n
        self.alpha = torch.tensor(alpha).float()
        self.sigma_rec = torch.tensor(sigma_rec).float()
        self.weight_decay = weight_decay
        self.input_size = 6
        self.recurrent_layer = torch.nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data =torch.zeros(self.n, self.n).float().to(device=device)

        self.activation=torch.nn.ReLU(inplace=False)

        self.input_layer = torch.nn.Linear(6, self.n, bias=False)
        self.input_layer.weight.data = torch.cat((torch.eye(6),torch.zeros(2,6)),dim=0).float().to(device=device)
        self.input_layer.weight.requires_grad = True

        self.output_layer = torch.nn.Linear(self.N, 2, bias=False)
        self.output_layer.weight.data = torch.cat((torch.zeros(2,self.n-2),torch.eye(2)),dim=1).float().to(device=device)
        self.output_layer.weight.requires_grad = True

        self.A = torch.nn.Parameter(torch.rand(self.N, self.N,device=device), requires_grad=True)
        self.Q = (torch.eye(self.N, device=device) - (self.A - self.A.t()) / 2) @ torch.inverse(torch.eye(self.N,device=device) + (self.A - self.A.t()) / 2)
        self.q = self.Q[:self.n, :]
        self.q = self.q.to(device=device)


    def forward(self, u):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]
        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                              std=1).to(
            device=device)

        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)


        return   states, self.output_layer(states)


    def cayley_transform(self):
        skew = (self.A - self.A.t()) / 2
        skew = skew.to(device=device)
        eye = torch.eye(self.N).to(device=device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        self.q = o[:self.n, :]

# Callbacks
def r2_scorer(y_true, y_pred):
    y_true = to_tensor(y_true,device=device)
    y_pred = to_tensor(y_pred,device=device)
    var_y = torch.var(y_true, unbiased=False)

    return 1-F.mse_loss(y_pred, y_true,reduction='mean') / var_y

r2 = EpochScoring(scoring=make_scorer(r2_scorer), on_train=False)

def r2_x(net, X , y):
    y = to_tensor(y, device=device)
    x = y[:, :, :-2]
    q = net.module_.q.detach()
    xqtq = x @ q.t() @ q
    var_x = torch.var(x, unbiased=False)
    return 1 -net.criterion_(x, xqtq) / var_x


def r2_xqt(net, X, y):

    xbar = net.predict(X)
    xbar = to_tensor(xbar, device=device)
    y = to_tensor(y, device=device)
    x = y[:, :, :-2]
    q = net.module_.q.detach()
    xqt = x @ q.t()
    var_xqt = torch.var(xqt, unbiased=False)
    return 1 -net.criterion_(xqt, xbar) / var_xqt


def r2_z(net, X, y):
    xbar = to_tensor(net.predict(X), device=device)
    zbar = to_tensor(net.module_.output_layer(xbar), device=device)
    y = to_tensor(y, device=device)
    z = y[:, :, -2:]
    return 1 -net.criterion_(z, zbar) / torch.var(z, unbiased=False)


def rsquared(net, X, y):
    xbar = to_tensor(net.predict(X), device=device)
    zbar = to_tensor(net.module_.output_layer(xbar), device=device)
    x = torch.cat((xbar@ net.module_.q.detach(),zbar),dim=2)
    y = to_tensor(y, device=device)

    return 1 - net.criterion_(y, x) / torch.var(y, unbiased=False)


def L2_weight(net,X = None, y = None):
    return torch.mean(torch.pow(net.module_.recurrent_layer.weight, 2))



class LatentNet(NeuralNetRegressor):
    def __init__(self, constrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.constrained=constrained

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        xbar = y_pred[0]
        zbar = y_pred[1]

        x = torch.cat((xbar @ self.module_.q, zbar), dim=2)

        return self.criterion_(y_true, x) / torch.var(y_true, unbiased=False)


    def train_step(self, Xi, yi, **fit_params):

        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)

        # if self.history[-1, 'epoch']<self.max_diagonal_epoch:
        #     self.module_.input_layer.weight.data = self.module_.input_mask * torch.relu(self.module_.input_layer.weight.data)
        #     self.module_.output_layer.weight.data = self.module_.output_mask * torch.relu(self.module_.output_layer.weight.data)
        # else:
        #     self.module_.input_layer.weight.data = torch.relu(self.module_.input_layer.weight.data)
        #     self.module_.output_layer.weight.data = torch.relu(self.module_.output_layer.weight.data)
        if self.constrained:
            self.module_.input_layer.weight.data = self.module_.input_mask * torch.relu(
                self.module_.input_layer.weight.data)
            self.module_.output_layer.weight.data = self.module_.output_mask * torch.relu(self.module_.output_layer.weight.data)
            self.module_.output_layer.weight.data = torch.relu(
                self.module_.output_layer.weight.data)
        else:
            self.module_.input_layer.weight.data = torch.relu(
                self.module_.input_layer.weight.data)
            self.module_.output_layer.weight.data = torch.relu(
                self.module_.output_layer.weight.data)
        self.module_.cayley_transform()

        return step_accumulator.get_step()






