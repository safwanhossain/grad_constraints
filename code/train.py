import torch
from torchdiffeq import odeint
import torch.utils.data
import csv
from tqdm import tqdm

from network import OneParam as Network
from time_dynamics import TimeDynamics


class derivative_net(object):
    def __init__(self, x_dim, y_dim, initial_c, true_fcn, dataset, batch_size):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.true_fcn = true_fcn
        self.dataset = dataset
        self.x_i = initial_c[0]
        self.y_i = initial_c[1]
        self.batch_size = batch_size
        self.num_epochs = 100
        self.lr = 0.2
        self.gpu = False

        self.network = Network(self.x_dim, self.y_dim)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)
        if self.gpu:
            self.network = self.network.cuda()

        loss_file = "loss.csv"
        loss_csv_file = open(loss_file, mode='w')
        self.loss_csv_writer = csv.writer(loss_csv_file, delimiter=',')

    def train(self):
        self.evaluate(250, -1)
        rtol, atol = 0.001, 0.001
        for epoch in range(self.num_epochs):
            t = torch.linspace(.0, 1.0, 20).clone().detach().requires_grad_(True)

            size = self.dataset.dataset.__len__() // self.batch_size
            for iter, data in tqdm(enumerate(self.dataset), total=size):
                if iter == size:
                    break
                x_train, y_train = data

                # TODO: Make it purely batch based
                approx_y = torch.zeros(self.batch_size, self.y_dim)
                loss = 0
                for index, x_f in enumerate(x_train):
                    val = odeint(TimeDynamics(self.x_i, x_f, self.network), self.y_i, t, rtol, atol)[-1]
                    approx_y[index] = val
                    loss += (y_train[index] - val) ** 2

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def evaluate(self, num_samples, epoch=0):
        filename = "results_" + str(epoch) + ".csv"
        csv_file = open(filename, mode='w')
        csv_writer = csv.writer(csv_file, delimiter=',')

        num_samples = 250
        x_test = (-2 - 2) * torch.rand(num_samples, 1) + 2
        y_test = self.true_fcn(torch.transpose(x_test, 0, 1))

        # number of evaluation points for the path
        num_t = 7
        t = torch.linspace(.0, 1.0, num_t).clone().detach().requires_grad_(True)
        total_loss = 0
        y_vals = []
        loss = 0
        for index, x_f in enumerate(x_test):
            rtol, atol = 0.001, 0.001
            y = odeint(TimeDynamics(self.x_i, x_f, self.network), self.y_i, t, rtol, atol)[-1]
            y_vals.append(round(y.data.item(), 4))
            csv_writer.writerow([str(x_f.item()), str(y.item())])
            loss += (y_test[index] - y) ** 2

        loss = loss / num_samples
        self.loss_csv_writer.writerow(["loss", str(loss)])
        print("Loss for Epoch ", epoch, "is", loss)


def true_y(x):
    """This gives true data points for training.
       This is the absolute value function
    :param x:
    :return:
    """
    # print(f, "sum:{torch.sum(x, 0)}, max:{torch.max(x, 0)[0]}, abs:{torch.abs(x)[0]}")
    y = torch.abs(x)[0]  # torch.sum(torch.abs(x), 0)  #torch.max(x, 0)[0]
    # print("True Y Values: ", y)
    return y


def create_dataset(train_x, train_y, batch_size):
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
                                       shuffle=True, drop_last=True, pin_memory=True)


def main():
    x_dim = 1
    y_dim = 1
    num_samples = 256

    # generate samples between -2 and 2 using uniform distribution
    x_train = (-2 - 2) * torch.rand(num_samples, 1) + 2
    y_train = true_y(torch.transpose(x_train, 0, 1))
    dataset = create_dataset(x_train, y_train, 64)
    initial_c = (x_train[0], y_train[0])

    deriv_net = derivative_net(x_dim, y_dim, initial_c, true_y, dataset, 64)
    deriv_net.train()


if __name__ == "__main__":
    main()
