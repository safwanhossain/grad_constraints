from torch import nn
import torch


class OneParam(nn.Module):
    def __init__(self, xdim, ydim):
        """This module computes the dynamics at a point x. That is it return the Jacobian matrix
        where each element is dy_i/dx_j
        Output is a matrix of size ydim x xdim
        """
        # Use Learning rate of 0.5 and no momentum (for SGD)

        super(OneParam, self).__init__()
        self.W = nn.Parameter(torch.zeros(xdim, requires_grad=True))
        # self.b = nn.Parameter(torch.zeros(xdim, requires_grad=True))

    def forward(self, x):
        out = torch.tanh(self.W * x)
        return out


class Dynamics(nn.Module):
    def __init__(self, xdim, ydim, final_activation=None):
        """This module computes the dynamics at a point x. That is it return the Jacobian matrix
        where each element is dy_i/dx_j
        Output is a matrix of size ydim x xdim
        """
        # Haven't gotten this network to learn yet ...

        super(Dynamics, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.final_activation = final_activation

        # Layers
        self.d = 25
        self.linear_layers = nn.Sequential(
            nn.Linear(xdim, self.d),
            torch.nn.LeakyReLU(),

            #nn.Linear(self.d, self.d),
            #torch.nn.LeakyReLU(),

            nn.Linear(self.d, xdim * ydim)
        )

    def forward(self, x):
        out = self.linear_layers(x)
        #out = out.view(-1, self.ydim, self.xdim)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out.reshape(-1,)#self.ydim,)


def unit_test():
    xdim = 1
    ydim = 1
    dynamics = Dynamics(xdim, ydim)

    # 20 batches of input, each batch with 10 elements
    xbatch = torch.randn(1, xdim)
    ybatch = dynamics.forward(xbatch)
    assert (ybatch.shape == (1,))


if __name__ == "__main__":
    unit_test()
    print("UNIT TESTS PASSED")
