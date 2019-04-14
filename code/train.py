import torch
from torchdiffeq import odeint

from network import OneParam as Network
from time_dynamics import TimeDynamics

class derivative_net(object):
    def __init__(self, x_dim, y_dim, x_train, y_train):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_train = x_train
        self.y_train = y_train
        self.x_i = self.x_train[0]
        self.y_i = self.y_train[0]
        self.num_epochs = 100
        self.lr = 0.5
        self.betas = (0.5, 0.9)
        self.gpu = False
        
        self.network = Network(self.x_dim, self.y_dim)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)
        if self.gpu:
            self.network = self.network.cuda()

    def train(self):
        rtol, atol = 0.001, 0.001
        for epoch in range(self.num_epochs):
            total_loss = 0
            t = torch.linspace(.0, 1.0, 20).clone().detach().requires_grad_(True)
            approx_y = torch.zeros(self.y_train.shape[0], self.y_dim)
           
            loss = 0
            for index, x_f in enumerate(self.x_train):
                val = odeint(TimeDynamics(self.x_i, x_f, self.network), self.y_i, t, rtol, atol)[-1]
                approx_y[index] = val
                loss += (self.y_train[index] - val)**2

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            print("Loss for Epoch ", epoch, "is", loss)
            print("approx_y", approx_y)
            print("true_y", self.y_train)

    def evaluate(num_samples, true_function):
        num_samples = 25
        x_test = torch.randn(num_samples,self.x_dim)
        y_test = true_function(x_test) # needs to be fixed
        
        # number of evaluation points for the path
        num_t = 10
        t = torch.linspace(.0, 1.0, num_t).clone().detach().requires_grad_(True)
        total_loss = 0
        y_vals = []
        for index, x_f in enumerate(x_test):
            rtol, atol = 0.001, 0.001
            y = approx_y(x_i_ex, x_f, grad_y, parameters, y_i_ex, t, rtol, atol)
            y_vals.append(round(y.data.item(), 4))

        # should actually plot them
        print(y_test)
        print(y_vals)

        
def true_y(x):
    """This gives true data points for training.
       This is the absolute value function
    :param x:
    :return:
    """
    #print(f, "sum:{torch.sum(x, 0)}, max:{torch.max(x, 0)[0]}, abs:{torch.abs(x)[0]}")
    y = torch.abs(x)[0]  # torch.sum(torch.abs(x), 0)  #torch.max(x, 0)[0]
    #print("True Y Values: ", y)
    return y

def main():
    x_dim = 1
    y_dim = 1
    num_samples = 20

    x_train = torch.randn(num_samples,1)
    y_train = true_y(torch.transpose(x_train, 0, 1))

    deriv_net = derivative_net(x_dim, y_dim, x_train, y_train)
    deriv_net.train()

if __name__ == "__main__":
    main()



