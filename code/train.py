import torch
from torchdiffeq import odeint
import torch.utils.data
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from network import Dynamics
from time_dynamics import TimeDynamics, InverseTimeDynamics


class derivative_net(object):
    def __init__(self, initial_c, true_fcn, train_dataset, test_dataset, num_train, num_test, batch_size,
                 num_epochs, rtol, atol, save_name, network_choice, optimizer, do_inverse):
        # TODO: This should be replaced with parser.args
        self.true_fcn = true_fcn

        self.train_dataset = train_dataset
        self.num_train = num_train
        self.test_dataset = test_dataset
        self.num_test = num_test
        self.do_inverse = do_inverse

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
        self.lr = 0.01
        self.momentum = 0.0  # 0.9
        self.gpu = False
        self.results_dir = 'results/'

        self.network = network_choice
        self.optimizer = optimizer(self.network.parameters(), lr=self.lr)  # , lr=self.lr, momentum=self.momentum)
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
                # print(torch.abs(loss), self.atol + self.rtol * torch.max(torch.abs(batch_data[1])))
                if torch.abs(loss) < (self.atol + self.rtol * torch.max(torch.abs(batch_data[1]))):
                    self.atol /= 2.0
                    self.rtol /= 2.0
                    print(f"Decaying atol, rtol to {self.atol}")
            train_loss = self.evaluate_dataset(self.train_dataset, epoch)
            print(f"Epoch {epoch} train loss: {train_loss}")

        test_loss = self.evaluate_dataset(self.test_dataset, self.num_epochs, do_inverse=self.do_inverse)
        print(f"Final test loss: {test_loss}")

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
        # print("Average loss for Epoch ", epoch, "is", total_loss)
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

            csv_writer.writerow([str(x.item()), str(y.item()), str(y_pred.item()), str(x_pred)])

        return self.loss(y, y_pred)


def true_y_abs(x):
    """This gives true data points for training.
       This is the absolute value function
    :param x:
    :return:
    """
    y = torch.abs(x)[0]
    return y


def true_y_exp(x):
    """This gives true data points for training.
       This is the exponential function.
    :param x:
    :return:
    """
    y = torch.exp(x)[0]
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


def init_ax():
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
    return fig, ax


def setup_ax(ax):
    ax.legend(fancybox=True, borderaxespad=0.0, framealpha=0.0)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def plot_functions(test_name, num_epochs, rtol, atol, x_o):
    fig, ax = init_ax()

    initial_x, initial_y, initial_y_pred, initial_x_pred = load_log('results/' + test_name + '_graph_-1.csv')
    final_x, final_y, final_y_pred, final_x_pred = load_log(
        'results/' + test_name + '_graph_' + str(num_epochs) + '.csv')

    train_x, train_y, _, _ = load_log('results/' + test_name + '_graph_0.csv')

    ax.axvline(x_o, linestyle='--', color='k', label='Initial Conditions')

    # Plot training data
    ax.scatter(train_x, train_y, label='Training Data', c='k', s=mpl.rcParams['lines.markersize'] ** 2 * 1.25)

    # Plot the true function
    ax.plot(initial_x, initial_y, label='True Function', c='r')

    # Plot the initial function
    ax.plot(initial_x, initial_y_pred, label='Initial Learned Function', c='g')
    initial_y_error = np.abs(initial_y_pred) * rtol + atol
    ax.fill_between(initial_x, initial_y_pred + initial_y_error, initial_y_pred - initial_y_error,
                    color='g', alpha=0.2)

    # Plot the learned function
    ax.plot(final_x, final_y_pred, label='Final Learned Function', c='b')
    final_y_error = np.abs(final_y_pred) * rtol + atol
    ax.fill_between(final_x, final_y_pred + final_y_error, final_y_pred - final_y_error,
                    color='b', alpha=0.2)

    ax = setup_ax(ax)
    fig.savefig('results/' + test_name + '_plot_functions.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_inverse_functions(test_name, num_epochs, rtol, atol, x_o):
    fig, ax = init_ax()

    initial_x, initial_y, initial_y_pred, initial_x_pred = load_log('results/' + test_name + '_graph_-1.csv')
    final_x, final_y, final_y_pred, final_x_pred = load_log(
        'results/' + test_name + '_graph_' + str(num_epochs) + '.csv')

    train_x, train_y, _, _ = load_log('results/' + test_name + '_graph_0.csv')

    ax.axvline(true_y_exp(x_o), linestyle='--', color='k', label='Initial Conditions')

    # Plot training data
    ax.scatter(train_y, train_x, label='Training Data', c='k', s=mpl.rcParams['lines.markersize'] ** 2 * 1.25)

    # Plot the true function
    ax.plot(initial_y, initial_x, label='True Inverse Function', c='r')

    # Plot the inverse
    zipped_inverse = zip(final_y, final_x_pred)
    sorted_zipped_inverse = sorted(zipped_inverse)
    final_y = np.array([y for y, _ in sorted_zipped_inverse])
    final_x_pred = np.array([x_pred for _, x_pred in sorted_zipped_inverse])

    ax.plot(final_y, final_x_pred, label='Final Learned Inverse', c='y')
    final_x_error = np.abs(final_x_pred) * rtol + atol
    ax.fill_between(final_y, final_x_pred + final_x_error, final_x_pred - final_x_error,
                     color='y', alpha=0.2)

    ax = setup_ax(ax)
    fig.savefig('results/' + test_name + '_plot_inverse_functions.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def invertible_experiment_compute(num_epochs, rtol, atol, x_o):
    save_name = 'invertible'
    x_dim = 1
    y_dim = 1
    num_train = 20
    batch_size = num_train
    num_test = 100

    # Create training data
    x_train = torch.linspace(-1, 1, num_train).reshape(-1, 1)
    y_train = true_y_exp(torch.transpose(x_train, 0, 1))
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_o, true_y_exp(x_o))

    # Create testing data
    x_test = torch.linspace(-1.5, 1.5, num_test).reshape(-1, 1)
    y_test = true_y_exp(torch.transpose(x_test, 0, 1))
    test_dataset = create_dataset(x_test, y_test, batch_size)

    # Create the network and train
    network_choice = Dynamics(x_dim, y_dim)
    deriv_net = derivative_net(initial_c, true_y_exp, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name, network_choice,
                               optimizer=torch.optim.Adam, do_inverse=True)
    deriv_net.train()
    return deriv_net.rtol, deriv_net.atol


def lipschitz_experiment_compute(num_epochs, rtol, atol, x_o):
    save_name = 'lipschitz'
    x_dim = 1
    y_dim = 1
    num_train = 20
    batch_size = num_train
    num_test = 100

    # Create training data.
    x_train = torch.linspace(-1, 1, num_train).reshape(-1, 1)
    y_train = true_y_abs(torch.transpose(x_train, 0, 1))
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_o, true_y_abs(x_o))

    # Create testing data.
    x_test = torch.linspace(-1.5, 1.5, num_test).reshape(-1, 1)
    y_test = true_y_abs(torch.transpose(x_test, 0, 1))
    test_dataset = create_dataset(x_test, y_test, batch_size)

    # Train the network.
    network_choice = Dynamics(x_dim, y_dim, torch.tanh)
    deriv_net = derivative_net(initial_c, true_y_abs, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name, network_choice,
                               optimizer=torch.optim.Adam, do_inverse=False)
    deriv_net.train()
    return deriv_net.rtol, deriv_net.atol


def main(init_rtol, init_atol):
    rtol_final_lipschitz = rtol_final_invertible = init_rtol
    atol_final_lipschitz = atol_final_invertible = init_atol
    x_o = torch.tensor([0.0])

    # TODO: Should change to argparse
    print("Beginning Lipschitz experiments...")
    num_epochs_lipschitz = 100
    rtol_final_lipschitz, atol_final_lipschitz = lipschitz_experiment_compute(num_epochs_lipschitz, init_rtol, init_atol, x_o)
    plot_functions('lipschitz', num_epochs_lipschitz, rtol_final_lipschitz, atol_final_lipschitz, x_o)

    print("Beginning Invertibility experiments...")
    num_epochs_invertible = 100
    rtol_final_invertible, atol_final_invertible = invertible_experiment_compute(num_epochs_invertible, init_rtol, init_atol, x_o)
    plot_functions('invertible', num_epochs_invertible, rtol_final_invertible, atol_final_invertible, x_o)
    plot_inverse_functions('invertible', num_epochs_invertible, rtol_final_invertible, atol_final_invertible, x_o)


if __name__ == "__main__":
    print("Beginning experiments...")
    init_rtol, init_atol = 0.1, 0.1
    main(init_rtol, init_atol)
    print("Finished!")
