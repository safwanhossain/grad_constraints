import torch
from torchdiffeq import odeint
import torch.utils.data
import csv
from tqdm import tqdm
import numpy as np

from network import Dynamics as Network
from time_dynamics import TimeDynamics, InverseTimeDynamics


class derivative_net(object):
    def __init__(self, x_dim, y_dim, initial_c, true_fcn, train_dataset, test_dataset, num_train, num_test, batch_size,
                 num_epochs, rtol, atol, save_name):
        # TODO: This should be replaced with parser.args
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.true_fcn = true_fcn

        self.train_dataset = train_dataset
        self.num_train = num_train
        self.test_dataset = test_dataset
        self.num_test = num_test

        self.x_i = initial_c[0]
        self.y_i = initial_c[1]

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_name = save_name

        self.num_batches = self.train_dataset.dataset.__len__() // self.batch_size

        num_t = 20
        self.t = torch.linspace(.0, 1.0, num_t).clone().detach().requires_grad_(True)

        self.rtol = rtol
        self.atol = atol
        self.lr = 0.0001
        self.momentum = 0.0  # 0.9
        self.gpu = False
        self.results_dir = 'results/'

        self.network = Network(self.x_dim, self.y_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters())  # , lr=self.lr, momentum=self.momentum)
        if self.gpu:
            self.network = self.network.cuda()

        loss_file = self.results_dir + "loss.csv"
        loss_csv_file = open(loss_file, mode='w')
        self.loss_csv_writer = csv.writer(loss_csv_file, delimiter=',')

    def loss(self, y_pred, y):
        return (y_pred - y) ** 2

    def train(self):
        print(f"Initial test loss: {self.evaluate_dataset(self.test_dataset, -1)}")
        for epoch in range(self.num_epochs):
            for batch_num, batch_data in enumerate(self.train_dataset):  # tqdm(, total=self.num_batches):

                # TODO: Make it purely batch based
                loss = self.evaluate_batch(batch_data)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                print(torch.abs(loss), self.atol + self.rtol * torch.max(torch.abs(batch_data[1])))
                if torch.abs(loss) < self.atol + self.rtol * torch.max(torch.abs(batch_data[1])):
                    self.atol /= 2
                    self.rtol /= 2
                    print(f"Decaying atol, rtol to {self.atol}")
            train_loss = self.evaluate_dataset(self.train_dataset, epoch)
            print(f"\nEpoch {epoch} train loss: {train_loss}")

        print(f"Final test loss: {self.evaluate_dataset(self.test_dataset, self.num_epochs, do_inverse=True)}")

    def evaluate_dataset(self, dataset, epoch=0, do_inverse=False):
        filename = self.results_dir + self.save_name + "_graph_" + str(epoch) + ".csv"
        csv_file = open(filename, mode='w')
        csv_writer = csv.writer(csv_file, delimiter=',')

        total_loss = float('inf')
        for batch_num, batch_data in enumerate(dataset):  # tqdm(, total=self.num_batches):
            batch_loss = self.evaluate_batch(batch_data, csv_writer, do_inverse)
            if batch_num == 0:
                total_loss = batch_loss
            else:
                total_loss += batch_loss

        total_loss = total_loss / self.num_batches
        self.loss_csv_writer.writerow(["loss", str(total_loss)])
        print("Average loss for Epoch ", epoch, "is", total_loss)
        return total_loss

    def evaluate_batch(self, batch_data, csv_writer=None, do_inverse=False):
        batch_x, batch_y = batch_data

        loss = float('inf')
        for batch_index, x_f in enumerate(batch_x):
            point_loss = self.evaluate_point(x_f, batch_y[batch_index], csv_writer, do_inverse)
            if batch_index == 0:
                loss = point_loss
            else:
                loss += point_loss
        return loss / self.batch_size

    def evaluate_point(self, x, y, csv_writer=None, do_inverse=False):
        y_pred = odeint(TimeDynamics(self.x_i, x, self.network), self.y_i, self.t, self.rtol, self.atol)[-1]
        if csv_writer is not None:
            x_pred = 0
            if do_inverse:
                x_pred = odeint(InverseTimeDynamics(self.y_i, y, self.network),
                                self.x_i, self.t, self.rtol, self.atol)[-1]
                x_pred = x_pred[0].item()
                print(f"y = {y}, x_pred = {x_pred}")

            csv_writer.writerow([str(x.item()), str(y.item()), str(y_pred.item()), str(x_pred)])

        return self.loss(y, y_pred)


def true_y(x):
    """This gives true data points for training.
       This is the absolute value function
    :param x:
    :return:
    """
    # print(f, "sum:{torch.sum(x, 0)}, max:{torch.max(x, 0)[0]}, abs:{torch.abs(x)[0]}")
    # y = torch.abs(x)[0]  # torch.sum(torch.abs(x), 0)  #torch.max(x, 0)[0]
    y = torch.exp(x)[0]
    # print("True Y Values: ", y)
    return y


def create_dataset(train_x, train_y, batch_size):
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)


def load_log(log_loc):
    data = {'x': [], 'y': [], 'y_pred': [], 'x_pred': []}
    with open(log_loc, 'r') as f:
        for line in f:
            components = line.split(',')
            assert len(components) == 4
            data['x'] += [float(components[0])]
            data['y'] += [float(components[1])]
            data['y_pred'] += [float(components[2])]
            data['x_pred'] += [float(components[3])]

    zipped_data = zip(data['x'], data['y'], data['y_pred'], data['x_pred'])
    sorted_zipped_data = sorted(zipped_data)
    x = np.array([x for x, _, _, _ in sorted_zipped_data])
    y = np.array([y for _, y, _, _ in sorted_zipped_data])
    y_pred = np.array([y_pred for _, _, y_pred, _ in sorted_zipped_data])
    x_pred = np.array([x_pred for _, _, _, x_pred in sorted_zipped_data])
    return (x, y, y_pred, x_pred)


def plot_functions(test_name, num_epochs, rtol, atol):
    initial_x, initial_y, initial_y_pred, initial_x_pred = load_log('results/' + test_name + '_graph_-1.csv')
    final_x, final_y, final_y_pred, final_x_pred = load_log(
        'results/' + test_name + '_graph_' + str(num_epochs) + '.csv')

    train_x, train_y, _, _ = load_log('results/' + test_name + '_graph_0.csv')

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set some parameters.
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    mpl.rcParams['legend.fontsize'] = 25
    mpl.rcParams['axes.labelsize'] = 25
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    mpl.rcParams['axes.grid'] = True

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(train_x, train_y, label='Train', c='k', s=mpl.rcParams['lines.markersize'] ** 2 * 1.25)

    ax.plot(initial_x, initial_y, label='True', c='r')

    ax.plot(initial_x, initial_y_pred, label='Initial', c='g')
    initial_y_error = np.abs(initial_y_pred) * rtol + atol
    ax.fill_between(initial_x, initial_y_pred + initial_y_error, initial_y_pred - initial_y_error,
                    color='g', alpha=0.2)

    ax.plot(final_x, final_y_pred, label='Final', c='b')
    final_y_error = np.abs(final_y_pred) * rtol + atol
    ax.fill_between(final_x, final_y_pred + final_y_error, final_y_pred - final_y_error,
                    color='b', alpha=0.2)

    zipped_inverse = zip(final_y, final_x_pred)
    sorted_zipped_inverse = sorted(zipped_inverse)
    final_y = np.array([y for y, _ in sorted_zipped_inverse])
    final_x_pred = np.array([x_pred for _, x_pred in sorted_zipped_inverse])

    ax.plot(final_x_pred, final_y, label='Final Inverse', c='y')
    final_x_error = np.abs(final_x_pred) * rtol + atol
    ax.fill_betweenx(final_y, final_x_pred + final_x_error, final_x_pred - final_x_error,
                     color='y', alpha=0.2)

    ax.legend(fancybox=True, borderaxespad=0.0, framealpha=0.0)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('results/' + test_name + '_plot_functions.png', bbox_inches='tight', dpi=200)


def main(save_name, num_epochs, rtol, atol):
    # TODO: Should change to argparse
    x_dim = 1
    y_dim = 1
    num_train = 20
    batch_size = num_train  # // 2
    num_test = 100

    # generate samples between -2 and 2 using uniform distribution
    scale = 3.0
    shift = scale / 2.0
    x_train = torch.rand(num_train, 1) * scale - shift
    y_train = true_y(torch.transpose(x_train, 0, 1))
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_train[0], y_train[0])

    x_test = torch.rand(num_test, 1) * scale - shift
    y_test = true_y(torch.transpose(x_test, 0, 1))
    test_dataset = create_dataset(x_test, y_test, batch_size)

    deriv_net = derivative_net(x_dim, y_dim, initial_c, true_y, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name=save_name)
    deriv_net.train()
    return deriv_net.rtol, deriv_net.atol

    # TODO: Need to plot true_y and predicted y over time?


if __name__ == "__main__":
    save_name = 'test_1'
    num_epochs = 100
    rtol, atol = 1.0, 1.0
    rtol, atol = main(save_name, num_epochs, rtol, atol)
    plot_functions('test_1', num_epochs, rtol, atol)
