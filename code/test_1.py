from torchdiffeq import odeint
from torch import nn
import torch
from scipy.integrate import odeint
import numpy as np


def interpolate(t, x_i, x_f):
    """

    :param t:
    :param x_i:
    :param x_f:
    :return:
    """
    # TODO: Change this to change our interpolation paths... Ex. add gaussian noise?
    return t * x_i + (1 - t) * x_f


class dynamics_basic(nn.Module):
    # the underlying function we are trying to learn is y = x^2. Here we are explicitly writing the derivative
    # dy/dx = 2x. Thus y is a 1d tensor and x is as usual 1d
    def __init__(self, x_0_i, x_1_i, x_0_f, x_1_f):
        """

        :param x_0_i:
        :param x_1_i:
        :param x_0_f:
        :param x_1_f:
        """
        self.x_0_i, self.x_1_i = x_0_i, x_1_i
        self.x_0_f, self.x_1_f = x_0_f, x_1_f
        self.x_0, self.x_1 = self.x_0_i, self.x_1_i
        # TODO: I don't like the module having a "state" that changes as we call forward, because side-effects bad...
        #   Is there any way around this?
        super(dynamics_basic, self).__init__()

    def forward(self, arg, t):
        """

        :param arg:
        :param t:
        :return:
        """
        # y_0 = arg[0]
        self.x_0, self.x_1 = interpolate(t, self.x_0_i, self.x_0_f), interpolate(t, self.x_1_i, self.x_1_f)

        # TODO: Justify this step.  Ex. by the FTLI this computes our path integral, which gives correct answer by FTC.
        gradient_term = [self.x_1, self.x_0]  # TODO: Make this a function, which is passed to integrate basic.
        path_rectification_term = [[self.x_0_i - self.x_0_f], [self.x_0_i - self.x_0_f]]
        #  TODO: We should probably have interpolate return the rectification term, incase it dynamically changes direction.
        y_increment = np.dot(gradient_term, path_rectification_term)  # TODO: Can I use @ notation for dot?
        print(f"y_increment: {y_increment}, x_0: {self.x_0}, x_1: {self.x_1}, t: {t}")
        return y_increment


def integrate_basic(num_t):
    """

    :param num_t:
    :return:
    """
    with torch.no_grad():
        y_0 = 0
        x_0_i, x_1_i = 0, 0
        x_0_f, x_1_f = 2, 2
        initial_val = [y_0]
        t = np.linspace(.0, 1.0, num_t)[::-1]  # torch.linspace(0., 1.0, num_t)[::-1]
        true_y = odeint(dynamics_basic(x_0_i, x_1_i, x_0_f, x_1_f), initial_val, t)
    return true_y


# TODO: parameterize the initial and final points.  Add arg to integrate_base
# TODO: rename variables and functions to make sense for this case
# TODO: non-linear paths?  Change interpolate(...) to do this.

# TODO: Parameterize the gradient function
# TODO: Try to learn the gradient function for some given (x_0, x_1, y_0) training tuples
# TODO: Compare learning gradient and integrating to directly regressing the function.

# TODO: Visualize the problem.
#   Ex. two axes - one for x-domain and one for y-domain
#   Plot the path in the x-domain, and associated y-gradients, and y_increments ( = gradient * rectifier)

if __name__ == "__main__":
    num_t = 1000
    sol = integrate_basic(num_t)
    print(sol[-1])
