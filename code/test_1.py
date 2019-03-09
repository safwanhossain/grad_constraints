from torchdiffeq import odeint
from torch import nn
import torch
from scipy.integrate import odeint
import numpy as np

def interpolate(t, x_i, x_f):
    return t*x_i + (1 - t)*x_f

class dynamics_basic(nn.Module):
    # the underlying function we are trying to learn is y = x^2. Here we are explicitly writing the derivative
    # dy/dx = 2x. Thus y is a 1d tensor and x is as usual 1d
    def __init__(self, x_0_i, x_1_i, x_0_f, x_1_f):
        self.x_0_i, self.x_1_i = x_0_i, x_1_i
        self.x_0_f, self.x_1_f = x_0_f, x_1_f
        self.x_0, self.x_1 = self.x_0_i, self.x_1_i
        super(dynamics_basic, self).__init__()

    def forward(self, arg, t):
        y_0 = arg[0]
        #  x_0, x_1 = arg[1], arg[2]

        self.x_0, self.x_1 = interpolate(t, self.x_0_i, self.x_0_f), interpolate(t, self.x_1_i, self.x_1_f)

        y_summer = np.dot([self.x_1, self.x_0], [[self.x_0_i - self.x_0_f], [self.x_0_i - self.x_0_f]])[0]
        # TODO: is it y_0 + y_summer?
        # TODO: is new_x_0 the value, or the difference?
        return_val = [y_summer]
        print(return_val, self.x_0, self.x_1, t)
        return return_val

def integrate_basic(num_t):
    with torch.no_grad():
        y_0 = 0
        x_0_i, x_1_i = 0, 0
        # x_0, x_1 = x_0_i, x_1_i
        x_0_f, x_1_f = 2, 2
        initial_val = [y_0]
        t = np.linspace(.0, 1.0, num_t)[::-1]  # torch.linspace(0., 1.0, num_t)[::-1]
        true_y = odeint(dynamics_basic(x_0_i, x_1_i, x_0_f, x_1_f), initial_val, t)
    return true_y


if __name__ == "__main__":
    num_t = 1000
    sol = integrate_basic(num_t)
    # sol = use_scipy_solver()
    print(sol)
