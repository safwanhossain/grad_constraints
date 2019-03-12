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
dot = np.dot
from scipy.integrate import odeint'''

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

    # TODO: I don't like the module having a "state" that changes as we call forward, because side-effects bad...
    #   Is there any way around this?
    def __init__(self, x_i, x_f, grad_y, parameters):
        """

        :param x_i:
        :param x_f:
        :param grad_y:
        :param W:
        """
        self.grad_y = grad_y
        self.x_i, self.x_f = x_i, x_f
        self.x = self.x_i
        self.Ws = [parameters[0]]
        self.bs = [parameters[1]]

        def interpolate(t):
            # TODO: Change this to change our interpolation paths... Ex. add gaussian noise?
            return (1.0 - t) * self.x_i + t * self.x_f

        self.interpolate = interpolate
        super(DynamicsBasic, self).__init__()

    def forward(self, t, arg):  # arg, t):
        """

        :param arg:
        :param t:
        :return:
        """
        # y_0 = arg[0]

        # First compute dy / dx for integration
        gradient_term = self.grad_y(self.x, self.Ws[0], self.bs[0])

        # Second, compute dx / dt now for rectification of integration.
        self.x = self.interpolate(t)
        path_rectification_term = torch.zeros(self.x.shape[0], dtype=dtype)
        for index, entry in enumerate(self.x):
            #entry.backward(create_graph=True, retain_graph=True)
            path_rectification_term[index] = torch.autograd.grad(entry, t, retain_graph=True)[0]  # t.grad
            # t.grad.data.zero_()
        #print(f"gradient_term: {gradient_term}, rect_term: {path_rectification_term}, y:{arg}")

        # Finally, combine dy/dx and dx/dt.
        y_increment = gradient_term @ path_rectification_term
        # print(f"y_increment: {y_increment}, x: {self.x}, t: {t}")
        return y_increment


def integrate_basic(num_t, y, grad_y, x_i, y_i_ex, x_train, y_train, parameters):
    """

    :param num_t:
    :param y:
    :param grad_y:
    :param x_i:
    :param y_i_ex:
    :param x_train:
    :param y_train:
    :param W:
    :return:
    """
    t = tensor_type(np.linspace(.0, 1.0, num_t), dtype=dtype, requires_grad=True)
    total_loss = 0
    for index, x_f in enumerate(x_train):
        rtol, atol = 0.01, 0.01
        approx_y = odeint(DynamicsBasic(x_i, x_f, grad_y, parameters), y_i_ex, t, rtol, atol)[-1]
        total_loss += (y_train[index] - approx_y) ** 2
    #print(approx_ys, y_train)

    print(f"Total loss: {total_loss}")
    return total_loss / float(x_train.shape[0])

def y_ex(x):
    """This gives true data points for training.

    :param x:
    :return:
    """
    return x[0] * x[1]


def grad_y_ex(x, W, b):
    """This gives true gradients for comparison.

    :param x:
    :param W: unused
    :return:
    """
    return tensor_type([x[1], x[0]], dtype=dtype)


def approx_grad_y(x, W, b):
    """This is our learned gradient.

    :param x:
    :param W:
    :return:
    """
    return W @ x + b  # dot(W, x)


def approx_y(x, W):
    """This is our learned function, computed by integrating the approximate derivative.

    :param x:
    :param W:
    :return:
    """
    pass  # TODO: Put ODE_int inside of here?


if __name__ == "__main__":
    torch.manual_seed(0)

    # Create a dataset.
    num_train = 10
    x_dim, y_dim = 2, 1
    x_train = torch.randn(num_train, x_dim)
    y_train = y_ex(torch.transpose(x_train, 0, 1))

    # Create our initial value.
    x_i_ex = tensor_type([.0, .0], dtype=dtype)
    y_i_ex = y_ex(x_i_ex)
    num_t_ex = 10

    # Create our initial weights
    W = torch.zeros(x_dim, x_dim, requires_grad=True)
    b = torch.zeros(x_dim, requires_grad=True)
    parameters = (W, b)
    # TODO: Should be a function that takes in x_dim and outputs y_dim x x_dim
    W.retain_grad = True
    b.retain_grad = True  # TODO: Do I need this?

    def curried_integrate_basic(parameters):
        # Swap grad_y_ex for approx_grad_y
        # return integrate_basic(num_t_ex, y_ex, grad_y_ex, x_i_ex, y_i_ex, x_train, y_train, W_cur)
        return integrate_basic(num_t_ex, y_ex, approx_grad_y, x_i_ex, y_i_ex, x_train, y_train, parameters)

    num_iters = 100
    lr = 0.1
    decay_val = 0.1
    for i in range(num_iters):
        pred_loss = curried_integrate_basic(parameters)
        reg_loss = decay_val * (torch.sum(torch.abs(W)) + torch.sum(torch.abs(b)))
        # print(reg_loss)
        total_loss = pred_loss + reg_loss
        print(f"Iteration:{i}, total_loss: {total_loss}, pred_loss: {pred_loss}, reg_loss: {reg_loss}")
        total_loss.backward()
        #print(W.grad, b.grad)
        for parameter in parameters:
            #parameter_grad = torch.autograd.grad(total_loss, parameter, retain_graph=True)[0]
            if parameter.grad is not None:
                parameter.data -= lr * parameter.grad
                parameter.grad.data.zero_()
        print(f"Final W: {W}, final b: {b}")


    # TODO:
    #   Make W the parameters to a deep net, as opposed to linear regression
    #   Train on minibatches of x?
    #   Make it work on learning high-dimensional y's
    #   Make it work on MNIST (high dim x and y)
    #   Non-linear interpolate?
    #   rename variables / functions to make sense now
    #   Make a way to visualize the problem:
    #       ex. x domain with path, y-domain with jabobian*rectifier field, y-domain with labels
    #   Compare learned gradient with exact gradient for regressing function?  Gets rid of integration numerical error
    #   Make it work on the GPU
    #   Write the README.
    #   Abstract away components which can be re-used?  Think about overall goals...
