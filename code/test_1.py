"""
Notes on this file
"""

from torch import nn
import torch
import numpy as np
from torchdiffeq import odeint

tensor_type = torch.tensor
dtype = torch.float32
dot = torch.dot

'''tensor_type = np.array
dtype = np.float64
dot = np.dot'''

# from scipy.integrate import odeint


'''# TODO: Should I pass this as a parameter to the integrate function?
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


def grad_interpolate(t, x_i, x_f):
    """To rectify our paths in integrate.

    :param t:
    :param x_i:
    :param x_f:
    :return:
    """
    # TODO: This function should be automatically generated from interpolate via autograd
    return -x_i + x_f'''


class DynamicsBasic(nn.Module):
    """
    Write something here. The name DynamicsBasic should probably be changed.
    """

    def __init__(self, x_i, x_f, grad_y):
        """

        :param x_i:
        :param x_f:
        :param grad_y:
        """
        self.grad_y = grad_y
        self.x_i, self.x_f = x_i, x_f
        self.x = self.x_i
        # TODO: I don't like the module having a "state" that changes as we call forward, because side-effects bad...
        #   Is there any way around this?
        super(DynamicsBasic, self).__init__()

    def forward(self, t, arg):  # arg, t):
        """

        :param arg:
        :param t:
        :return:
        """
        # y_0 = arg[0]

        # First compute dy / dx for integration
        gradient_term = self.grad_y(self.x)  # TODO: Make this a function, which is passed to integrate basic.

        # Second, compute dx / dt now for rectification of integration.
        t.requires_grad = True

        def interpolate(t_local):
            # TODO: Change this to change our interpolation paths... Ex. add gaussian noise?
            return (1.0 - t_local) * self.x_i + t_local * self.x_f

        self.x = interpolate(t)
        path_rectification_term = torch.zeros(self.x.shape[0], dtype=dtype)
        for index, entry in enumerate(self.x):
            entry.backward(create_graph=True, retain_graph=True)
            path_rectification_term[index] = t.grad
            t.grad.data.zero_()
        #path_rectification_term = torch.transpose(torch.tensor([-self.x_i + self.x_f], dtype=dtype), 0 , 1)  #
        print(f"gradient_term: {gradient_term}, rect_term: {path_rectification_term}, y:{arg}")

        # Finally, combine dy/dx and dx/dt.
        # TODO: Justify this step.  Ex. by the FTLI this computes our path integral, which gives correct answer by FTC.
        y_increment = gradient_term @ path_rectification_term # TODO: Can I use @ notation for dot?
        print(f"y_increment: {y_increment}, x: {self.x}, t: {t}")
        return y_increment


def integrate_basic(num_t, y, grad_y, x_i, y_i_ex, x_train, y_train):
    """

    :param num_t:
    :param y:
    :param grad_y:
    :param x_i:
    :param x_f:
    :return:
    """
    #y_f = tensor_type([y(x_f)], dtype=dtype)

    t = tensor_type(np.linspace(.0, 1.0, num_t), dtype=dtype)
    approx_ys = torch.tensor([odeint(DynamicsBasic(x_i, x_f, grad_y), y_i_ex, t)[-1] for x_f in x_train], dtype=dtype)
    print(approx_ys, y_train)
    losses = (y_train - approx_ys) ** 2  # [(y_train - approx_y) ** 2 for y_index, approx_y in enumerate(approx_ys)]
    total_loss = torch.sum(losses) / float(len(losses))
    print(f"Total loss: {total_loss}")
    return total_loss  #calculated_y, y_f


# TODO: rename variables and functions to make sense for this case
# TODO: non-linear paths?  Change interpolate(...) to do this.

# TODO: Parameterize the gradient function
# TODO: Try to learn the gradient function for some given (x_0, x_1, y_0) training tuples
# TODO: Compare learning gradient and integrating to directly regressing the function.

# TODO: Visualize the problem.
#   Ex. two axes - one for x-domain and one for y-domain
#   Plot the path in the x-domain, and associated y-gradients, and y_increments ( = gradient * rectifier)

def y_ex(x):
    """This gives true data points for training.

    :param x:
    :return:
    """
    return x[0] * x[1]


def grad_y_ex(x):
    """This gives true gradients for comparison.

    :param x:
    :return:
    """
    return tensor_type([x[1], x[0]], dtype=dtype)

def approx_grad_y(x, W):
    """This is our learned gradient.

    :param x:
    :param W:
    :return:
    """
    return dot(W, x)

def approx_y(x, W):
    """This is our learned function, computed by integrating the approximate derivative.

    :param x:
    :param W:
    :return:
    """


if __name__ == "__main__":
    torch.manual_seed(0)

    # TODO: Create a dataset.
    num_train = 1
    x_dim = 2
    x_train = torch.randn(num_train, x_dim)
    y_train = y_ex(torch.transpose(x_train, 0, 1))  # [y_ex(x) for x in x_train]
    print(x_train)
    print(y_train)
    #print("Hello")

    x_i_ex = tensor_type([.0, .0], dtype=dtype)  # TODO: Pass the intial y-value too?
    y_i_ex = y_ex(x_i_ex)
    #x_f_ex = tensor_type([2.0, 2.0], dtype=dtype)  # TODO: replaced by training set
    num_t_ex = 10
    total_loss = integrate_basic(num_t_ex, y_ex, grad_y_ex, x_i_ex, y_i_ex, x_train, y_train)
    #approx_y, true_y = sol

    ## TODO: This is basically my loss!  Define grad_y_ex with parameters that optimized
    #print(f"Calculated y = {approx_y[-1]}, true y = {true_y}, difference: {approx_y[-1][0] - true_y[0]}")
