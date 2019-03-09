from torchdiffeq import odeint
from torch import nn
import torch
from scipy.integrate import odeint
import numpy as np


class dynamics_parametric(nn.Module):
    # the underlying function we are trying to learn is y = x^2. In this example, we are expressing it
    # parametrically in terms of t. Therefore, eq of parabola becomes [x, y] = [t, t^2] and it's 
    # derivative is [dx/dt, dy/dt] = [1, 2t]. Therefore, y is a tensor of dim 2, and t as usual is 1d
    def forward(self, y, t):
        return [1, 2 * t]


class dynamics_basic(nn.Module):
    # the underlying function we are trying to learn is y = x^2. Here we are explicitly writing the derivative
    # dy/dx = 2x. Thus y is a 1d tensor and x is as usual 1d
    def forward(self, y, x):
        return 2 * x


def integrate_parametric():
    # derivative is [dx/dt, dy/dt] = [1, 2t]
    with torch.no_grad():
        initial_val = torch.tensor([0, 0])
        t = torch.linspace(0., 6., 100)
        true_y = odeint(dynamics_parametric(), initial_val, t)
    return true_y


def integrate_basic():
    with torch.no_grad():
        initial_val = torch.tensor([0])
        t = torch.linspace(0., 6., 100)
        true_y = odeint(dynamics_basic(), initial_val, t)
    return true_y


def diff_eq(y, x):
    return [1, 2 * x]


def use_scipy_solver():
    # this works
    sol = odeint(diff_eq, [0, 0], np.linspace(0, 5, 100))
    return sol


if __name__ == "__main__":
    sol = integrate_basic()
    # sol = use_scipy_solver()
    print(sol)
