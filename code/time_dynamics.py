from torchdiffeq import odeint
from torch import nn
import torch

dtype = torch.float32

class TimeDynamics(nn.Module):
    """
    Our goal is to compute y(x_f) = y(x_0) + \int{y'(c(t))c'(t)}. To do so, we'll use the odeint
    for that need to model y'(c(t))c'(t), or the time dynamics. This module for the time dynamics
    """

    def __init__(self, x_i, x_f, deriv_y):
        """
        :param x_i: initial x value
        :param x_f: final x value
        :param grad_y: network to compute Jacobian
        """
        self.deriv_y = deriv_y
        self.x_i, self.x_f = x_i, x_f

        def path(t):
            """ For now, consider a straight line between x_i and x_f
            :param t:
            :return:
            """
            # Add a way to easily parametrize other paths
            return (1.0 - t) * self.x_i + t * self.x_f

        self.path = path
        super(TimeDynamics, self).__init__()

    def forward(self, t, y):
        """

        :param t: array of time instances at which to compute dynamics for
        :return: dy/dt = dy/dx*dx/dt
        """
        # compute the Input dynamics dy/dx
        curr_x = self.path(t)
        gradient_term = self.deriv_y(curr_x)
        
        # compute path dynamics dx/dt
        path_derivative = torch.zeros(curr_x.shape[0], dtype=dtype)
        for index, entry in enumerate(curr_x):
            path_derivative[index] = torch.autograd.grad(entry, t, retain_graph=True)[0]  # t.grad
        #path_derivative = torch.autograd.grad(curr_x, t, retain_graph=True)[0].reshape(1,) 
        
        # Finally, combine dy/dx and dx/dt to get dy/dt
        y_increment = gradient_term @ path_derivative
        return y_increment


def unit_test_1_1():
    """ Test for the learned function mapping from R to R
        Function is y = x^2
    """
    def test_dynamics(x):
        # x is a scalar
        return 2*x
    
    x_vals = torch.tensor([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]).requires_grad_(True)
    x_i = torch.tensor([0.0]).requires_grad_(True)
    y_i = torch.tensor([0.0]).requires_grad_(True)
    t = torch.linspace(.0, 1.0, 20).clone().detach().requires_grad_(True)
    rtol, atol = 0.001, 0.001

    approx_y = torch.zeros(x_vals.shape[0])
    for index, val in enumerate(x_vals):
        approx_y[index] = odeint(TimeDynamics(x_i, val, test_dynamics), y_i, t, rtol, atol)[-1]
    real_y = torch.pow(x_vals, 2)
    assert(torch.sum(torch.pow(approx_y-real_y, 2)).data < 0.0001)

def unit_test_n_1():
    """ Test for the learned function mapping from R^n to R
        Function is y = x_1^2 + x_2^2
    """
    
    def test_dynamics(x):
        # x in an R^n vector
        return 2*x.data
    
    x_vals = torch.tensor([[0,0], [0,1], [1,0], [1,2], [1,2], [3,2], [4,5]]).float().requires_grad_(True)
    x_i = torch.tensor([0.0, 0.0]).requires_grad_(True)
    y_i = torch.tensor([0.0]).requires_grad_(True)
    t = torch.linspace(.0, 1.0, 20).clone().detach().requires_grad_(True)
    rtol, atol = 0.001, 0.001

    approx_y = torch.zeros(x_vals.shape[0])
    for index, val in enumerate(x_vals):
        approx_y[index] = odeint(TimeDynamics(x_i, val, test_dynamics), y_i, t, rtol, atol)[-1]
    real_y = torch.sum(torch.pow(x_vals, 2), dim=1)
    assert(torch.sum(torch.pow(approx_y-real_y, 2)).data < 0.0001)

def unit_test_n_n():
    """ Test for the learned function mapping from R^n to R^n
        Function is y_1 = x_1^2 + x_2^2
                    y_2 = x_1 + x_2
    """
    
    def test_dynamics(x):
        # output should be a matrix
        return torch.cat([2*x.data, torch.ones(2)]).reshape(2,2)
    
    x_vals = torch.tensor([[0,0], [0,1], [1,0], [1,2], [1,2], [3,2], [4,5]]).float().requires_grad_(True)
    x_i = torch.tensor([0.0, 0.0]).requires_grad_(True)
    y_i = torch.tensor([0.0, 0.0]).requires_grad_(True)
    t = torch.linspace(.0, 1.0, 20).clone().detach().requires_grad_(True)
    rtol, atol = 0.001, 0.001

    approx_y = torch.zeros(x_vals.shape[0], 2)
    for index, val in enumerate(x_vals):
        approx_y[index] = odeint(TimeDynamics(x_i, val, test_dynamics), y_i, t, rtol, atol)[-1]
    real_y = torch.transpose(torch.stack([torch.sum(torch.pow(x_vals, 2), dim=1), torch.sum(x_vals, dim=1)]), 0, 1)
    assert(torch.sum(torch.pow(approx_y-real_y, 2)).data < 0.0001)

if __name__ == "__main__":
    unit_test_1_1()
    unit_test_n_1()
    unit_test_n_n() 
    print("UNIT TESTS PASSED")




