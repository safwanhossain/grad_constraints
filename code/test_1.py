"""
Notes on this file
"""

from torch import nn
import torch

import numpy as np


tensor_type = torch.tensor
dtype = torch.float64
dot = torch.dot
from torchdiffeq import odeint

'''tensor_type = np.array
dtype = np.float64
dot = np.dot'''
# from scipy.integrate import odeint



# TODO: Should I pass this as a parameter to the integrate function?
def interpolate(t, x_i, x_f):
    """To compute paths for integrate.

    :param t:
    :param x_i:
    :param x_f:
    :return:
    """
    # TODO: Change this to change our interpolation paths... Ex. add gaussian noise?
    # TODO: Return the rectificaiton term.  Currently [x_i - x_f], but depends on ?
    # print(x_i, x_f, (1.0 - t) * x_i + t * x_f, t)
    return (1.0 - t) * x_i + t * x_f
    #return ((t) * x_i + (1.0 - t) * x_f)


def grad_interpolate(t, x_i, x_f):
    """To rectify our paths in integrate.

    :param t:
    :param x_i:
    :param x_f:
    :return:
    """
    # TODO: This function should be automatically generated from interpolate via autograd
    return -x_i + x_f
    #return (x_i - x_f)


class DynamicsBasic(nn.Module):
    """
    Write something here. The name DynamicsBasic should probably be changed.
    """

    def __init__(self, x_0_i, x_1_i, x_0_f, x_1_f, grad_y):
        """

        :param x_0_i:
        :param x_1_i:
        :param x_0_f:
        :param x_1_f:
        :param grad_y:
        """
        self.grad_y = grad_y
        self.x_0_i, self.x_1_i = x_0_i, x_1_i
        self.x_0_f, self.x_1_f = x_0_f, x_1_f
        self.x_0, self.x_1 = self.x_0_i, self.x_1_i
        # TODO: I don't like the module having a "state" that changes as we call forward, because side-effects bad...
        #   Is there any way around this?
        super(DynamicsBasic, self).__init__()

    def forward(self, t, arg): #arg, t):
        """

        :param arg:
        :param t:
        :return:
        """
        # y_0 = arg[0]

        self.x_0, self.x_1 = interpolate(t, self.x_0_i, self.x_0_f), interpolate(t, self.x_1_i, self.x_1_f)

        # TODO: Justify this step.  Ex. by the FTLI this computes our path integral, which gives correct answer by FTC.
        gradient_term = self.grad_y(self.x_1, self.x_0)  # TODO: Make this a function, which is passed to integrate basic.
        path_rectification_term = tensor_type([grad_interpolate(t, self.x_0_i, self.x_0_f),
                                                grad_interpolate(t, self.x_1_i, self.x_1_f)],
                                               dtype=dtype)
        print(f"gradient_term: {gradient_term}, rect_term: {path_rectification_term}")
        #  TODO: We should probably have interpolate return the rectification term, incase it dynamically changes direction.
        y_increment = tensor_type([dot(gradient_term, path_rectification_term)],
                                   dtype=dtype) # TODO: Can I use @ notation for dot?
        print(f"y_increment: {y_increment}, x_0: {self.x_0}, x_1: {self.x_1}, t: {t}")
        return y_increment


def integrate_basic(num_t, y, grad_y, x_0_i, x_1_i, x_0_f, x_1_f):
    """

    :param num_t:
    :param y:
    :param grad_y:
    :param x_0_i:
    :param x_1_i:
    :param x_0_f:
    :param x_1_f:
    :return:
    """
    y_i = tensor_type([y(x_0_i, x_1_i)], dtype=dtype)
    y_f = tensor_type([y(x_0_f, x_1_f)], dtype=dtype)

    arr = np.linspace(.0, 1.0, num_t)
    t = tensor_type(arr, dtype=dtype) # torch.linspace(0., 1.0, num_t)[::-1]
    with torch.no_grad():
        calculated_y = odeint(DynamicsBasic(x_0_i, x_1_i, x_0_f, x_1_f, grad_y), y_i, t)
    return calculated_y, y_f


# TODO: parameterize the initial and final points.  Add arg to integrate_base
# TODO: rename variables and functions to make sense for this case
# TODO: non-linear paths?  Change interpolate(...) to do this.

# TODO: Parameterize the gradient function
# TODO: Try to learn the gradient function for some given (x_0, x_1, y_0) training tuples
# TODO: Compare learning gradient and integrating to directly regressing the function.

# TODO: Visualize the problem.
#   Ex. two axes - one for x-domain and one for y-domain
#   Plot the path in the x-domain, and associated y-gradients, and y_increments ( = gradient * rectifier)

def y_ex(x_0, x_1):
    """

    :param x_0:
    :param x_1:
    :return:
    """
    return x_0 * x_1


def grad_y_ex(x_0, x_1):
    """This is used as a parameter to the integrate function

    :param x_0:
    :param x_1:
    :return:
    """
    # TODO: This function should be automatically generated from y_example via autograd
    return tensor_type([x_1, x_0], dtype=dtype)


if __name__ == "__main__":
    num_t_ex = 10
    x_0_i_ex, x_1_i_ex = 1, 1  # TODO: Generalize this to be an arbitrary tensor.
    x_0_f_ex, x_1_f_ex = 2, 2
    sol = integrate_basic(num_t_ex, y_ex, grad_y_ex,
                          x_0_i_ex, x_1_i_ex, x_0_f_ex, x_1_f_ex)
    approx_y, true_y = sol
    print(f"Calculated y = {approx_y[-1]}, true y = {true_y}, difference: {approx_y[-1][0] - true_y[0]}")
    # TODO: Change everything to be torch.tensor, so works with ricky's ode int!!!
