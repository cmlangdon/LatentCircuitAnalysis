
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


class RegressionModule(torch.nn.Module):
    def __init__(self, orthogonal, N):
        super(RegressionModule, self).__init__()
        self.n = 4
        self.N = N
        self.orthogonal = orthogonal

        if self.orthogonal:
            self.A = torch.nn.Parameter(torch.rand(self.N, self.N,device=device), requires_grad=True)
            self.Q = (torch.eye(self.N, device=device) - (self.A - self.A.t()) / 2) @ torch.inverse(torch.eye(self.N,device=device) + (self.A - self.A.t()) / 2)
            self.q = self.Q[:self.n, :]
            self.q = self.q.to(device=device)
        else:
            self.q = torch.nn.Parameter(torch.rand(self.n, self.N, device=device), requires_grad=True)
    def forward(self, u):

        y_pred = u @ self.q

        return y_pred

    def cayley_transform(self):
        skew = (self.A - self.A.t()) / 2
        skew = skew.to(device=device)
        eye = torch.eye(self.N).to(device=device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        self.q = o[:self.n, :]



class RegressionNet(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, Xi, yi, **fit_params):

        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)

        if self.module_.orthogonal:
            self.module_.cayley_transform()

        return step_accumulator.get_step()







