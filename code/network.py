from torch import nn
import torch

class Dynamics(nn.Module):
    def __init__(self, xdim, ydim):
        """This module computes the dynamics at a point x. That is it return the Jacobian matrix
        where each element is dy_i/dx_j
        Output is a matrix of size ydim x xdim
        """
        super(Dynamics, self).__init__()
        self.xdim = xdim
        self.ydim = ydim

        # Layers
        self.d = 32
        self.linear_layers = nn.Sequential(
            nn.Linear(xdim, self.d),
            nn.LeakyReLU(),

            nn.Linear(self.d, self.d*2),
            nn.LeakyReLU(),

            nn.Linear(self.d*2, self.d),
            nn.LeakyReLU(),

            nn.Linear(self.d, 50)
        )

    def forward(self, x):
        batch_size = x.shape[1]
        out = self.linear_layers(x)
        out = out.view(-1, batch_size, self.ydim, self.xdim)
        return out

def unit_test():
    xdim = 5
    ydim = 10
    dynamics = Dynamics(xdim, ydim)

    # 20 batches of input, each batch with 10 elements
    xbatch = torch.randn(20, 10, 5)
    ybatch = dynamics.forward(xbatch)
    assert(ybatch.shape == (20, 10, 10, 5))

if __name__ == "__main__":
    unit_test()
    print("UNIT TESTS PASSED")
    

