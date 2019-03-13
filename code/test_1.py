"""
Notes on this file
"""
from torch import nn
import torch
import torch.autograd as autograd
from torch.nn import Sigmoid
from torchdiffeq import odeint

dtype = torch.float32


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
        :param parameters:
        """
        self.grad_y = grad_y
        self.x_i, self.x_f = x_i, x_f
        self.x = self.x_i
        self.Ws = [parameters[0]]
        self.bs = [parameters[1]]

        def interpolate(t):
            """

            :param t:
            :return:
            """
            # TODO: Change this to change our interpolation paths... Ex. add gaussian noise?
            return (1.0 - t) * self.x_i + t * self.x_f

        self.interpolate = interpolate
        super(DynamicsBasic, self).__init__()

    def forward(self, t, y):
        """

        :param arg:
        :param y:
        :return:
        """
        # First compute dy / dx for integration
        gradient_term = self.grad_y(self.x, self.Ws[0], self.bs[0])

        # Second, compute dx / dt now for rectification of integration.
        self.x = self.interpolate(t)
        path_rectification_term = torch.zeros(self.x.shape[0], dtype=dtype)
        for index, entry in enumerate(self.x):
            path_rectification_term[index] = autograd.grad(entry, t, retain_graph=True)[0]  # t.grad
        # print(f"gradient_term: {gradient_term}, rect_term: {path_rectification_term}, y:{arg}")

        # Finally, combine dy/dx and dx/dt.
        y_increment = gradient_term @ path_rectification_term
        # print(f"y_increment: {y_increment}, x: {self.x}, t: {t}")
        return y_increment


def integrate_basic(num_t, y, grad_y, x_i, y_i, x_train, y_train, parameters):
    """

    :param num_t:
    :param y:
    :param grad_y:
    :param x_i:
    :param y_i:
    :param x_train:
    :param y_train:
    :param parameters:
    :return:
    """
    t = torch.linspace(.0, 1.0, num_t).clone().detach().requires_grad_(True)
    total_loss = 0
    for index, x_f in enumerate(x_train):
        rtol, atol = 0.001, 0.001
        y = approx_y(x_i, x_f, grad_y, parameters, y_i, t, rtol, atol)
        total_loss += (y_train[index] - y) ** 2

    # print(f"Total loss: {total_loss}")
    return total_loss / float(x_train.shape[0])


def y_ex(x):
    """This gives true data points for training.

    :param x:
    :return:
    """
    print(f"sum:{torch.sum(x, 0)}, max:{torch.max(x, 0)[0]}, abs:{torch.abs(x)[0]}")
    return torch.abs(x)[0]  # torch.sum(torch.abs(x), 0)  #torch.max(x, 0)[0]


def grad_y_ex(x, W, b):
    """

    :param x:
    :param W:
    :param b:
    :return:
    """
    return torch.tensor([x[1], x[0]], dtype=dtype)


def approx_grad_y(x, W, b):
    """

    :param x:
    :param W:
    :param b:
    :return:
    """
    # TODO: some deep neural network of x.
    return Sigmoid()(W @ x) * 2.0 - 1.0


def approx_y(x_i, x_f, grad_y, parameters, y_i, t, rtol, atol):
    """

    :param x_i:
    :param x_f:
    :param grad_y:
    :param parameters:
    :param y_i:
    :param t:
    :param rtol:
    :param atol:
    :return:
    """
    # TODO: I can apply an activation on our final value too.
    return odeint(DynamicsBasic(x_i, x_f, grad_y, parameters), y_i, t, rtol, atol)[-1]


def main(args):
    """

    :param args:
    :return:
    """
    # Create a dataset.
    num_train = 10
    x_dim, y_dim = 1, 1
    x_train = torch.randn(num_train, x_dim)
    y_train = y_ex(torch.transpose(x_train, 0, 1))

    # Create our initial value.
    x_i_ex = torch.zeros(x_dim, dtype=dtype)
    y_i_ex = y_ex(x_i_ex)
    num_t_ex = 10

    # Create our initial weights
    W = torch.zeros(x_dim, x_dim, requires_grad=True)
    b = torch.zeros(x_dim, requires_grad=True)
    parameters = (W, b)  # TODO: Should be a function that takes in x and outputs (dim(y), dim(x)) Jacobian.

    def curried_integrate_basic(parameters_loc):
        """

        :param parameters_loc:
        :return:
        """
        # Swap grad_y_ex for approx_grad_y
        # return integrate_basic(num_t_ex, y_ex, grad_y_ex, x_i_ex, y_i_ex, x_train, y_train, W_cur)
        return integrate_basic(num_t_ex, y_ex, approx_grad_y, x_i_ex, y_i_ex, x_train, y_train, parameters_loc)

    pred_loss = 10e32  # Some initial value
    num_iters = 100  # The number of gradient descent iterations.
    lr = 1.0  # 0.1  # The learning rate for gradient descent.
    decay_val = 0.0  # 1  # The multiplier for our weight decay.
    for i in range(num_iters):
        pred_loss = curried_integrate_basic(parameters)
        reg_loss = decay_val * (torch.sum(torch.abs(W)) + torch.sum(torch.abs(b)))
        # print(reg_loss)
        total_loss_cur = pred_loss + reg_loss
        print(f"Iteration:{i}, total_loss: {total_loss_cur}, pred_loss: {pred_loss}, reg_loss: {reg_loss}")
        total_loss_cur.backward()
        # print(W.grad, b.grad)
        for parameter in parameters:
            # parameter_grad = torch.autograd.grad(total_loss, parameter, retain_graph=True)[0]
            if parameter.grad is not None:
                parameter.data -= lr * parameter.grad
                parameter.grad.data.zero_()
        print(f"Final W: {W}, final b: {b}, final prediction loss: {pred_loss}")


if __name__ == "__main__":
    torch.manual_seed(0)
    args = None  # TODO: Add argument parser
    main(args)

    # TODO:
    #   Type annotations
    #   Make a validation set and compare with train loss
    #   Create logging for a grapher - losses and all values
    #   compute integral from closest "query" point?
    #   Understand how to set rtol/atol
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
