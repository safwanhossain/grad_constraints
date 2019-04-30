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
        self.lr = 0.005
        self.momentum = 0.0  # 0.9
        self.gpu = False
        self.results_dir = 'results/'

        self.network = network_choice
        self.optimizer = optimizer(self.network.parameters(), lr=self.lr)  # , lr=self.lr, momentum=self.momentum)
        if self.gpu:
            self.network = self.network.cuda()

        loss_file = self.results_dir + self.save_name + "_loss.csv"
        loss_csv_file = open(loss_file, mode='w')
        self.loss_csv_writer = csv.writer(loss_csv_file, delimiter=',')

    def loss(self, y_pred, y):
        return torch.sum(torch.abs(y_pred - y))

    def train(self):
        with torch.no_grad():
            print(f"Initial test loss: {self.evaluate_dataset(self.test_dataset, -1, write_csv=False)}")
            print(f"Epoch {0} train loss: {self.evaluate_dataset(self.train_dataset, 0, write_csv=True)}")
        for epoch in range(self.num_epochs):
            for batch_num, batch_data in enumerate(self.train_dataset):  # tqdm(, total=self.num_batches):

                # TODO: Make it purely batch based
                loss = self.evaluate_batch(batch_data)

                self.optimizer.zero_grad()
                loss.backward()  # retain_graph=True)
                self.optimizer.step()
                #print(batch_data[1])
                y_mag = torch.max(torch.sum(torch.abs(batch_data[1]), dim=0))
                self.loss_csv_writer.writerow([str(torch.abs(loss).item()), self.atol, self.rtol, str(y_mag.item())])
                if torch.abs(loss) < (self.atol + self.rtol * y_mag):
                    self.atol /= 2.0
                    self.rtol /= 2.0
                    print(f"Decaying atol, rtol to {self.atol}")
            with torch.no_grad():
                train_loss = self.evaluate_dataset(self.train_dataset, epoch + 1, write_csv=True)
                print(f"Epoch {epoch} train loss: {train_loss}")

        with torch.no_grad():
            test_loss = self.evaluate_dataset(self.test_dataset, self.num_epochs, do_inverse=self.do_inverse,
                                              write_csv=False)
            print(f"Final test loss: {test_loss}")

    def evaluate_dataset(self, dataset, epoch=0, do_inverse=False, write_csv=False):
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
            x_pred = torch.zeros(x.shape[0])
            if do_inverse:
                x_pred = odeint(InverseTimeDynamics(self.y_i, y, self.network),
                                self.x_i, self.t, self.rtol, self.atol)[-1]
                #  x_pred = x_pred[0].item()
            csv_writer.writerow([str([val.item() for val in x])[1:-1],
                                 str([val.item() for val in y])[1:-1],
                                 str([val.item() for val in y_pred])[1:-1],
                                 str([val.item() for val in x_pred])[1:-1]])
            # csv_writer.writerow([str(x.item()), str(y.item()), str(y_pred.item()), str(x_pred)])

        return self.loss(y, y_pred)


def create_dataset(train_x, train_y, batch_size):
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)


def load_log_plot(log_loc):
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


def load_log_loss(log_loc):
    data = {'losses': [], 'rtols': [], 'atols': [], 'y_mags': []}
    with open(log_loc, 'r') as f:
        for line in f:
            components = line.split(',')
            assert len(components) == 4
            data['losses'] += [float(components[0])]
            data['atols'] += [float(components[1])]
            data['rtols'] += [float(components[2])]
            data['y_mags'] += [float(components[3])]

    return np.array(data['losses']), np.array(data['rtols']), np.array(data['atols']), np.array(data['y_mags'])


def init_ax():
    # Set some parameters.
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    fontsize = 16
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.grid'] = True

    fig = plt.figure(figsize=(6.4, 4.8))
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


def plot_loss(test_name):
    fig, ax = init_ax()
    losses, rtols, atols, y_mags = load_log_loss('results/' + test_name + '_loss.csv')

    iterations = range(len(losses))
    ax.semilogy(iterations, losses, c='r', label='Training Loss')
    ax.semilogy(iterations, atols + rtols * y_mags, c='g', label='Tolerance')

    ax = setup_ax(ax)
    fig.savefig('images/' + test_name + '_plot_losses.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    return rtols[-1], atols[-1]


def plot_functions(test_name, num_epochs, rtol, atol, x_o, true_y):
    fig, ax = init_ax()

    initial_x, initial_y, initial_y_pred, initial_x_pred = load_log_plot('results/' + test_name + '_graph_-1.csv')
    final_x, final_y, final_y_pred, final_x_pred = load_log_plot(
        'results/' + test_name + '_graph_' + str(num_epochs) + '.csv')

    baseline_x, baseline_y, baseline_y_pred, baseline_x_pred = load_log_plot(
        'results/' + test_name + '_incorrect_graph_' + str(num_epochs) + '.csv')

    train_x, train_y, _, _ = load_log_plot('results/' + test_name + '_graph_0.csv')

    # Plot the true function
    ax.plot(initial_x, initial_y, label='True Function', c='r')

    # Plot the initial function
    ax.plot(initial_x, initial_y_pred, label='Initial Learned Function', c='g')
    initial_y_error = np.abs(initial_y_pred) * rtol + atol
    ax.fill_between(initial_x, initial_y_pred + initial_y_error, initial_y_pred - initial_y_error,
                    color='g', alpha=0.2)

    # Plot the learned function
    ax.plot(final_x, final_y_pred, label='Final Constrained Learned Function', c='b')
    final_y_error = np.abs(final_y_pred) * rtol + atol
    ax.fill_between(final_x, final_y_pred + final_y_error, final_y_pred - final_y_error,
                    color='b', alpha=0.2)

    ax.plot(baseline_x, baseline_y_pred, label='Final Unconstrained Learned Function', c='m')
    baseline_y_error = np.abs(baseline_y_pred) * rtol + atol
    ax.fill_between(baseline_x, baseline_y_pred + baseline_y_error, baseline_y_pred - baseline_y_error,
                    color='m', alpha=0.2)

    ax.scatter(x_o, true_y(x_o), color='y', marker='*', label='Initial Conditions',
               s=mpl.rcParams['lines.markersize'] * 8, zorder=20)

    #ax.set_ylim(min(initial_y), max(initial_y))

    # Plot training data
    ax.scatter(train_x, train_y, label='Training Data', c='k',
               s=mpl.rcParams['lines.markersize'] * 2, zorder=10)

    ax = setup_ax(ax)
    fig.savefig('images/' + test_name + '_plot_functions.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def true_y_abs(x):
    """This gives true data points for training.
       This is the absolute value function
    :param x:
    :return:
    """
    y = torch.abs(x)
    return y


def true_y_exp(x):
    """This gives true data points for training.
       This is the exponential function.
    :param x:
    :return:
    """
    y = torch.exp(x)#torch.sigmoid(x * 5) * 2 #exp(x)
    return y


def plot_inverse_functions(test_name, num_epochs, rtol, atol, x_o):
    fig, ax = init_ax()

    initial_x, initial_y, initial_y_pred, initial_x_pred = load_log_plot('results/' + test_name + '_graph_-1.csv')
    final_x, final_y, final_y_pred, final_x_pred = load_log_plot(
        'results/' + test_name + '_graph_' + str(num_epochs) + '.csv')

    baseline_x, baseline_y, baseline_y_pred, baseline_x_pred = load_log_plot(
        'results/' + test_name + '_incorrect_graph_' + str(num_epochs) + '.csv')

    train_x, train_y, _, _ = load_log_plot('results/' + test_name + '_graph_0.csv')

    # Plot the true function
    ax.plot(initial_y, initial_x, label='True Inverse Function', c='r')

    # Plot the inverse
    zipped_inverse = zip(final_y, final_x_pred)
    sorted_zipped_inverse = sorted(zipped_inverse)
    final_y = np.array([y for y, _ in sorted_zipped_inverse])
    final_x_pred = np.array([x_pred for _, x_pred in sorted_zipped_inverse])

    ax.plot(final_y, final_x_pred, label='Final Constrained Learned Inverse', c='b')
    final_x_error = np.abs(final_x_pred) * rtol + atol
    ax.fill_between(final_y, final_x_pred + final_x_error, final_x_pred - final_x_error,
                    color='b', alpha=0.2)

    ax.plot(baseline_y, baseline_x_pred, label='Final Unconstrained Learned Inverse', c='m')
    baseline_x_error = np.abs(baseline_x_pred) * rtol + atol
    ax.fill_between(baseline_y, baseline_x_pred + baseline_x_error, baseline_x_pred - baseline_x_error,
                    color='m', alpha=0.2)

    ax.scatter(true_y_exp(x_o), x_o, color='y', marker='*', label='Initial Conditions',
               s=mpl.rcParams['lines.markersize'] * 8, zorder=20)

    # Plot training data
    # ax.scatter(train_y, train_x, label='Training Data', c='k', s=mpl.rcParams['lines.markersize'])

    ax = setup_ax(ax)
    fig.savefig('images/' + test_name + '_plot_inverse_functions.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def invertible_experiment_compute(num_epochs, rtol, atol, x_o, activation, save_name):
    x_dim = 1
    y_dim = 1
    num_train = 5
    batch_size = num_train
    num_test = 101

    # Create training data
    x_train = torch.linspace(-1, 1, num_train).reshape(-1, x_dim)
    target_function = true_y_exp
    y_train = target_function(x_train)
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_o, target_function(x_o))

    # Create testing data
    x_test = torch.linspace(-2, 2, num_test).reshape(-1, x_dim)
    y_test = target_function(x_test)
    test_dataset = create_dataset(x_test, y_test, batch_size)

    network_choice = Dynamics(x_dim, y_dim, activation)
    deriv_net = derivative_net(initial_c, target_function, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name, network_choice,
                               optimizer=torch.optim.Adam, do_inverse=True)
    deriv_net.train()


def lipschitz_experiment_compute(num_epochs, rtol, atol, x_o, activation, save_name):
    x_dim = 1
    y_dim = 1
    num_train = 5
    batch_size = num_train
    num_test = 101

    # Create training data.
    x_train = torch.linspace(-1, 1, num_train).reshape(-1, x_dim)
    target_function = true_y_abs
    y_train = target_function(x_train)
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_o, target_function(x_o))

    # Create testing data.
    x_test = torch.linspace(-2.0, 2.0, num_test).reshape(-1, x_dim)
    y_test = target_function(x_test)
    test_dataset = create_dataset(x_test, y_test, batch_size)

    # Train the network.
    network_choice = Dynamics(x_dim, y_dim, activation)
    deriv_net = derivative_net(initial_c, target_function, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name, network_choice,
                               optimizer=torch.optim.Adam, do_inverse=False)
    deriv_net.train()


def high_dim_invertible_experiment_compute(num_epochs, rtol, atol):
    save_name = 'high_dim_invertible'
    x_dim = 100  # 2
    y_dim = 100  # TODO: Should be dynamically set from the target function
    num_train = 20
    batch_size = 1  # num_train // 4  #num_train
    num_test = num_train

    x_o = torch.zeros(x_dim)
    # Create training data
    x_train = torch.randn(num_train, x_dim)
    target_function = true_y_exp
    y_train = target_function(x_train)
    train_dataset = create_dataset(x_train, y_train, batch_size)
    initial_c = (x_o, target_function(x_o))

    # Create testing data
    x_test = torch.randn(num_train, x_dim) * 1.5  # torch.linspace(-1.5, 1.5, num_test).reshape(-1, 1)
    y_test = target_function(x_test)
    test_dataset = create_dataset(x_test, y_test, batch_size)

    # Create the network and train
    epsilon = 0.0001

    def square(x):
        val = x @ torch.transpose(x, 0, 1) + epsilon * torch.eye(x_dim, y_dim)
        return val  # + epsilon * torch.ones(x_dim, y_dim)

    network_choice = Dynamics(x_dim, y_dim, square)
    deriv_net = derivative_net(initial_c, target_function, train_dataset, test_dataset, num_train, num_test,
                               batch_size, num_epochs, rtol, atol, save_name, network_choice,
                               optimizer=torch.optim.Adam, do_inverse=True)
    deriv_net.train()


def main(init_rtol, init_atol):
    x_o = torch.zeros(1)

    # TODO: Should change to argparse
    '''print("Beginning Lipschitz experiments...")
    num_epochs_lipschitz = 50
    lipschitz_experiment_compute(num_epochs_lipschitz, init_rtol, init_atol, x_o, torch.tanh, 'lipschitz')
    def scale_tanh(x): return x  #torch.tanh(x) * 10.0
    lipschitz_experiment_compute(num_epochs_lipschitz, init_rtol, init_atol, x_o, scale_tanh, 'lipschitz_incorrect')
    rtol_final_lipschitz, atol_final_lipschitz = plot_loss('lipschitz')
    plot_functions('lipschitz', num_epochs_lipschitz, rtol_final_lipschitz, atol_final_lipschitz, x_o, true_y_abs)'''

    print("Beginning Invertibility experiments...")
    num_epochs_invertible = 50
    def square(x):
        val = x ** 2 + 0.001
        return val
    invertible_experiment_compute(num_epochs_invertible, init_rtol, init_atol, x_o, square, 'invertible')
    def scale_tanh(x): return torch.tanh(x) * 10
    invertible_experiment_compute(num_epochs_invertible, init_rtol, init_atol, x_o, scale_tanh, 'invertible_incorrect')
    rtol_final_invertible, atol_final_invertible = plot_loss('invertible')
    plot_functions('invertible', num_epochs_invertible, rtol_final_invertible, atol_final_invertible, x_o, true_y_exp)
    plot_inverse_functions('invertible', num_epochs_invertible, rtol_final_invertible, atol_final_invertible, x_o)

    '''print("Beginning High Dimensionality Invertibility experiments...")
    num_epochs_invertible_high_dim = 100
    high_dim_invertible_experiment_compute(num_epochs_invertible_high_dim, init_rtol, init_atol)
    rtol_final_invertible, atol_final_invertible = plot_loss('high_dim_invertible')'''


if __name__ == "__main__":
    print("Beginning experiments...")
    init_rtol, init_atol = 0.1, 0.1
    main(init_rtol, init_atol)
    print("Finished!")
