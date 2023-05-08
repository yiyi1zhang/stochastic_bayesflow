import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as op
import numpy as np
import scipy.stats as st
from sklearn.metrics import r2_score
import statsmodels.api as sm
from tqdm import tqdm
import itertools
import FrEIA.framework as Ff
import FrEIA.modules as Fm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTNet_Encoder(nn.Module):

    def __init__(self, in_features, out_features, conv_out, conv_kernel, gru_out, skip_steps, reccs_out, ar_window):
        super(LSTNet_Encoder, self).__init__()
        self.num_features = in_features
        self.conv1_out_channels = conv_out
        self.conv1_kernel_height = conv_kernel
        self.recc1_out_channels = gru_out
        self.skip_steps = skip_steps
        self.skip_reccs_out_channels = reccs_out
        self.output_out_features = out_features
        self.ar_window_size = ar_window
        self.conv1 = nn.Conv2d(1, self.conv1_out_channels,
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.bn1 = nn.BatchNorm2d(self.conv1_out_channels)
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.skip_reccs = nn.ModuleList()
        for i in range(len(self.skip_steps)):
            self.skip_reccs.append(nn.GRU(self.conv1_out_channels, self.skip_reccs_out_channels[i], batch_first=True))
        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar1 = nn.Linear(self.num_features, self.output_out_features)
            self.ar = nn.Linear(self.ar_window_size, 1)

    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)

        # Convolutional Layer
        C = X.unsqueeze(1)  # [batch_size, num_channels=1, time_steps, num_features]
        C = self.conv1(C)
        C = self.bn1(F.relu(C))  # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = torch.squeeze(C, 3)  # [batch_size, conv1_out_channels, shrinked_time_steps]

        # Recurrent Layer
        R = C.permute(0, 2, 1)  # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R)  # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :]  # [batch_size, recc_out_channels]

        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip_steps)):
            skip_step = self.skip_steps[i]
            skip_sequence_len = shrinked_time_steps // skip_step
            # shrinked_time_steps shrinked further
            S = C[:, :, -skip_sequence_len * skip_step:]  # [batch_size, conv1_out_channels, shrinked_time_steps]
            S = S.view(S.size(0), S.size(1), skip_sequence_len,
                       skip_step)  # [batch_size, conv1_out_channels, skip_sequence_len, skip_step=num_skip_components]
            # note that num_skip_components = skip_step
            S = S.permute(0, 3, 2,
                          1).contiguous()  # [batch_size, skip_step=num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.view(S.size(0) * S.size(1), S.size(2),
                       S.size(3))  # [batch_size*num_skip_components, skip_sequence_len, conv1_out_channels]
            out, hidden = self.skip_reccs[i](
                S)  # [batch_size*num_skip_components, skip_sequence_len, skip_reccs_out_channels[i]]
            S = out[:, -1, :]  # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            S = S.view(batch_size,
                       skip_step * S.size(1))  # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            R = torch.cat((R, S), 1)  # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]
            # print(S.shape)
        # print(R.shape)

        # Output Layer
        O = self.output(R)  # [batch_size, output_out_features]
        # O = torch.cat((O, X[:, 0, :]), 1)
        if self.ar_window_size > 0:
            # set dim3 based on output_out_features
            AR = self.ar1(X[:, -self.ar_window_size:, :]) # [batch_size, ar_window_size, output_out_features]
            AR = AR.permute(0, 2, 1).contiguous()  # [batch_size, output_out_features, ar_window_size]
            AR = self.ar(AR)  # [batch_size, output_out_features, 1]
            AR = AR.squeeze(2)  # [batch_size, output_out_features]
            O = O + AR

        return O



class ConditionalINN(nn.Module):
    """combine conditional INN blocks (RNVP Copling Block) with conditional network"""

    def __init__(self, input_dim, cond_dim, summary_dim, conv_out, conv_kernel, gru_out, skip_steps, reccs_out,
                 ar_window, n_blocks, hidden_layer_size):
        super().__init__()
        self.input_dim = input_dim
        self.n_blocks = n_blocks
        self.cINN = self.build_cinn(input_dim, summary_dim, n_blocks, hidden_layer_size, exponent_clamping=1,
                                    clamp_activation='I')
        self.condNet = LSTNet_Encoder(cond_dim, summary_dim, conv_out, conv_kernel, gru_out, skip_steps, reccs_out, ar_window)

    def parameters(self):
        return itertools.chain(self.cINN.parameters(), self.condNet.parameters())

    def named_parameters(self):
        return itertools.chain(self.cINN.named_parameters(), self.condNet.named_parameters())

    def build_cinn(self, input_dim, summary_dim, n_blocks, hidden_layer_size, exponent_clamping=2.,
                   clamp_activation='TANH'):
        nodes = [Ff.InputNode(input_dim)]

        def CouplingNet(dropout):
            def _constructor(in_dim, out_dim):
                neurons = [hidden_layer_size, hidden_layer_size, hidden_layer_size, hidden_layer_size]
                if dropout:
                    layers = nn.ModuleList([nn.Linear(in_dim, neurons[0]),
                                            nn.Tanh(),
                                            Fm.MCDropout(dropout_rate)])
                else:
                    layers = nn.ModuleList([nn.Linear(in_dim, neurons[0]),
                                            nn.Tanh()])
                for i in range(1, len(neurons)):
                    layers.append(nn.Linear(neurons[i - 1], neurons[i]))
                    layers.append(nn.Tanh())
                    if dropout:
                        layers.append(Fm.MCDropout(dropout_rate))
                layers.append(nn.Linear(neurons[-1], out_dim))
                return nn.Sequential(*layers)

            return _constructor

        conditions = [Ff.ConditionNode(summary_dim)]
        for i in range(n_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.RNVPCouplingBlock,
                                 {'subnet_s_constructor': CouplingNet(dropout),
                                  'subnet_t_constructor': CouplingNet(dropout),
                                  'clamp': exponent_clamping,
                                  'clamp_activation': clamp_activation},
                                 conditions=conditions[0],
                                 name=f"coupling{i + 1}"))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': i}, name=f"Perm{i + 1}"))

        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + conditions, verbose=False)

    def forward(self, x, y):
        cond = self.condNet(y)
        z, jac = self.cINN(x, c=cond, jac=True)

        return [z], jac

    def sample(self, y, n_samples, device='cpu', dropout=True):
        z_samples = torch.randn(n_samples, y.shape[0], self.input_dim).to(device)
        cond = self.condNet(y)
        for _, layer in self.cINN.named_modules():
            if isinstance(layer, Fm.MCDropout):
                layer.set_active(dropout)
        x_samples = self.cINN(z_samples, c=torch.stack([cond] * n_samples), rev=True, jac=False)[0]

        return x_samples

    def read_params(self):
        params = []
        names = []
        for name, param in self.cinn.named_parameters():
            params.append(torch.sum(param.data).detach().cpu().numpy())
            names.append(name)
        return names, params

    def save(self, name):
        torch.save(
            {
                "net": self.cINN.state_dict(),
                "cond_net": self.condNet.state_dict()
            },
            name
        )

    def load(self, name):
        print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.cINN.load_state_dict(state_dicts["net"])
        self.condNet.load_state_dict((state_dicts["cond_net"]))
        self.cINN.eval()
        self.condNet.eval()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save(self.path)
        self.val_loss_min = val_loss


def load_data(file_theta, file_sig, split0=300, split1=300, split2=400, split3=500):
    """
    load MEMS simulation data
    split them into train, validation and test data
    :return: train_theta, train_signal, test_theta, test_signal
    """
    theta = np.load(file_theta)
    sig = np.load(file_sig)
    sig_c1 = sig[:, np.arange(0, sig.shape[1], 10), :]
    X_train, theta_train = sig_c1[:split0, :], theta[:split0, :]
    X_val, theta_val = sig_c1[split1:split2, :], theta[split1:split2, :]
    X_test, theta_test = sig_c1[split2:split3, :], theta[split2:split3, :]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    theta_val = torch.tensor(theta_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    theta_test = torch.tensor(theta_test, dtype=torch.float32)

    return X_train, theta_train, X_val, theta_val, X_test, theta_test


def maximum_likelihood_loss(z, log_det_J, weight=1.0):
    norm = 0.5 * torch.square(torch.norm(z, dim=-1))
    log = - log_det_J
    return torch.mean(norm + weight * log), torch.mean(norm), torch.mean(log)


def train_model(model, optimizer, data_generator, val_generator, iterations, lr_scheduler, ep):
    # training
    model.train(True)
    for it in range(iterations):
        X_batch, theta_batch = next(iter(data_generator))
        X_batch, theta_batch = X_batch.to(device), theta_batch.to(device)
        Z, log_det_J = model(theta_batch, X_batch)
        ml_loss, norm_loss, log_loss = maximum_likelihood_loss(Z[0], log_det_J, weight=1)
        ml_loss.retain_grad()
        optimizer.zero_grad()
        ml_loss.backward()
        optimizer.step()

    # validation
    model.train(False)
    for X_val, theta_val in iter(val_generator):
        X_val, theta_val = X_val.to(device), theta_val.to(device)
        Z_val, log_det_J_val = model(theta_val, X_val)
        ml_val, norm_val, log_val = maximum_likelihood_loss(Z_val[0], log_det_J_val, weight=1)

    if ep < 5 * max_explr_epoch:
        lr_scheduler.step()

    return ml_loss.item(), norm_loss.item(), log_loss.item(), Z, ml_val.item(), norm_val.item(), log_val.item()


def confidence_interval(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=np.std(data, axis=0))


def confidence_interval_sem(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=st.sem(data, axis=0))


def plot_inference_scatter_CI(theta_approx_flat, theta_test, n_samples, param_names, figsize=(10, 10), filename=None, n_columns=3):
    plt.rcParams['font.size'] = 12
    theta_test = theta_test.numpy()
    f, ax = plt.subplots(int(np.ceil(theta_dim/n_columns)), n_columns, figsize=figsize)
    if theta_dim == 1:
        ax = [ax]
    else:
        ax = ax.flat
    theta_approx_mean = theta_approx_flat.mean(axis=0)
    CI = confidence_interval(theta_approx_flat)
    for j in range(len(param_names)):
        ax[j].scatter(theta_test[:, j], theta_approx_mean[:, j], color='#5c92e8', s=40, alpha=0.8)
        lower_lim = min(ax[j].get_xlim()[0], ax[j].get_ylim()[0])
        upper_lim = max(ax[j].get_xlim()[1], ax[j].get_ylim()[1])
        ax[j].set_xlim((lower_lim, upper_lim))
        ax[j].set_ylim((lower_lim, upper_lim))
        ax[j].plot(ax[j].get_xlim(), ax[j].get_xlim(), '--', color='black')

        for i in range(len(theta_test)):
            ax[j].plot((theta_test[i, j], theta_test[i, j]), (CI[0][i, j], CI[1][i, j]), color='#5c92e8', alpha=0.5)
        rmse = np.sqrt(np.mean((theta_approx_mean[:, j] - theta_test[:, j]) ** 2))
        nrmse = rmse / (theta_test[:, j].max() - theta_test[:, j].min())
        r2 = r2_score(theta_test[:, j], theta_approx_mean[:, j])
        ax[j].text(0.1, 0.9, 'RMSE={:.3f}'.format(rmse), transform=ax[j].transAxes,
                   horizontalalignment='left', verticalalignment='center', size=10)
        # ax[j].text(0.1, 0.8, 'NRMSE={:.3f}'.format(nrmse), transform=ax[j].transAxes,
        #            horizontalalignment='left', verticalalignment='center', size=10)
        ax[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2), transform=ax[j].transAxes,
                   horizontalalignment='left', verticalalignment='center', size=10)
        if j == 0:
            ax[j].set_xlabel('True')
            ax[j].set_ylabel('Estimated')
        ax[j].set_title(param_names[j])
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
    if theta_dim > 1 and theta_dim % n_columns == n_columns - 1:
        ax[-1].set_visible(False)
    # f.tight_layout()
    plt.show()
    if filename is not None:
        f.savefig(filename)
        plt.close()


def plot_posterior_distribution(theta_approx_flat, theta_test, param_names, index, filepath=None, show=False):
    """
    plot the posterior distribution of the test sample id
    :param theta_approx: the output of the inverse model by sampling the latent space for n_samples times
    :type theta_approx: numpy array in the shape of (n_samples, n_test_samples, len(param_names))
    :param theta_test: the test parameters
    :type theta_test: numpy array in the shape of (n_test_samples, len(param_names))
    :param param_names: the selected parameter names
    :type param_names: List[str]
    :param index: the test sample id
    :type index: int
    :param filepath: the file path to store the plots
    :type filepath: str
    :param show: whether to show figure after storage, False or True
    :type show: bool
    """
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.size'] = 8
    if len(param_names) < 4:
        n_columns = len(param_names)
        n_rows = 1
    else:
        n_rows = int(np.sqrt(len(param_names)))
        n_columns = int(np.ceil(len(param_names)/n_rows))

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
    if len(param_names) == 1:
        ax = [ax]
    else:
        ax = ax.flat
    CI = confidence_interval(theta_approx_flat)
    for j in range(len(param_names)):
        sns.distplot(theta_approx_flat[:, index, j], ax=ax[j])
        ax[j].set_ylabel("")
        mean, stddev = st.norm.fit(theta_approx_flat[:, index, j])

        abs_rel_err = np.abs((mean - theta_test[index, j]) / theta_test[index, j])
        ax[j].axvspan(CI[0], CI[1], facecolor='g', alpha=0.25)
        ax[j].axvline(mean, color='#5c92e8', label='Estimated mean')
        ax[j].axvline(theta_test[index, j], color='#e55e5e', label=True)
        ax[j].text(0.1, 0.9, r"$\mu$"+"={:.3f}".format(mean), horizontalalignment='left',
                   verticalalignment='center', transform=ax[j].transAxes, size=10)
        ax[j].text(0.1, 0.8, r"$\sigma$"+"={:.3f}".format(stddev), horizontalalignment='left',
                   verticalalignment='center', transform=ax[j].transAxes, size=10)
        ax[j].text(0.1, 0.7, r"$ER$"+"={:.3f}".format(abs_rel_err), horizontalalignment='left',
                   verticalalignment='center', transform=ax[j].transAxes, size=10)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        if CI[0] < theta_test[index, j] < CI[1]:
            ax[j].set_title(f"{param_names[j]}" + "\n" + r"$\theta_{%s}$" % j + "={:.3f}".format(theta_test[index, j]) +
                            r"$\in$" + "[{0:.3f}, {1:.3f}]".format(CI[0], CI[1]), color='green')
        else:
            ax[j].set_title(f"{param_names[j]}" + "\n" + r"$\theta_{%s}$" % j + "={:.3f}".format(theta_test[index, j]) +
                            r"$\notin$" + "[{0:.3f}, {1:.3f}]".format(CI[0], CI[1]), color='red')

    if len(param_names) % n_columns == n_columns - 1:
        ax[-1].set_visible(False)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    filename = os.path.join(filepath, f'test_posterior_{index}.png')
    print(filename)
    plt.savefig(filename, bbox_inches='tight')
    if not show:
        plt.close()


def bh_dist(X1, X2):
    cX = np.concatenate([X1, X2])
    N = X1.shape[0]
    # Get density functions:
    kde_sk1 = sm.nonparametric.KDEUnivariate(X1)
    kde_sk1.fit()
    kde_sk2 = sm.nonparametric.KDEUnivariate(X2)
    kde_sk2.fit()
    # Calc coeff:
    xs = np.linspace(min(cX), max(cX), N)
    bht = 0
    for x in xs:
        y_sk1 = kde_sk1.evaluate(x)
        y_sk2 = kde_sk2.evaluate(x)
        bht += np.sqrt(y_sk1 * y_sk2) * (max(cX) - min(cX)) / N
    return -np.log(bht)


def plot_distribution_data_set(theta_approx, theta_test, filename=None, show=False):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.size'] = 8
    if len(param_names) < 4:
        n_columns = len(param_names)
        n_rows = 1
    else:
        n_rows = int(np.sqrt(len(param_names)))
        n_columns = int(np.ceil(len(param_names) / n_rows))

    theta_approx_mean = theta_approx.mean(axis=0)
    theta_test = theta_test.detach().numpy()

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
    if len(param_names) == 1:
        ax = [ax]
    else:
        ax = ax.flat
    for i in range(len(param_names)):

        for k in range(theta_approx.shape[0]):
            l = sns.kdeplot(theta_approx[k, :, i], ax=ax[i], color='#b3e6ff')
        l = sns.kdeplot(theta_approx_mean[:, i], ax=ax[i], color='#5c92e8')
        l = sns.kdeplot(theta_test[:, i], ax=ax[i], color='#e55e5e')
        ax[i].set_ylabel("")
        ax[i].set_title(param_names[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        if i == 0:
            ax[i].legend([l.get_lines()[-3], l.get_lines()[-2],l.get_lines()[-1]],
                         [r'$p_{est.}(\widehat{\theta}_{d}^{(k, b)})$', r'$p_{est.}(\widehat{\theta}_{d}^{(k)})$', r'$p_{true}(\theta_{d}^{(k)})}]$'])
    if theta_dim > 1 and theta_dim % n_columns == n_columns - 1:
        ax[-1].set_visible(False)
    plt.show()
    if filename is not None:
        print(filename)
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


def cicp_nmciw(theta_approx, theta_test, n_test):
    n_cicp = np.zeros(len(param_names))
    CI = confidence_interval(theta_approx)
    CI_sem = confidence_interval_sem(theta_approx)
    for j in range(len(param_names)):
        for i in range(n_test):
            if CI[0][i, j] < theta_test[i, j] < CI[1][i, j]:
                n_cicp[j] += 1
    CICP = 1 / n_test * (n_cicp)
    NMCIW = 1 / n_test * np.sum(CI[1] - CI[0], axis=0) / (np.max(theta_test, axis=0) - np.min(theta_test, axis=0))
    NMCIW_sem = 1 / n_test * np.sum(CI_sem[1] - CI_sem[0], axis=0) / (np.max(theta_test, axis=0) - np.min(theta_test, axis=0))
    return CICP, NMCIW, NMCIW_sem


def metrics_summary(theta_test, theta_test_approx, theta_test_approx_mean):
    rmse = np.sqrt(np.mean((theta_test_approx_mean - theta_test) ** 2, axis=0, keepdims=True))
    nrmse = rmse / (np.max(theta_test, axis=0) - np.min(theta_test, axis=0))
    r2 = [r2_score(theta_test[:, j], theta_test_approx_mean[:, j]) for j in range(len(param_names))]
    maxae = np.max(abs(theta_test - theta_test_approx_mean), axis=0)
    mae = np.mean(abs(theta_test - theta_test_approx_mean), axis=0)
    cicp, nmciw, nmciw_sem = cicp_nmciw(theta_test_approx, theta_test, n_test)
    nll = np.mean(0.5 * (np.log(np.var(theta_test_approx_mean, axis=0)) + (theta_test_approx_mean - theta_test) ** 2 /
                         np.var(theta_test_approx_mean, axis=0)), axis=0) + 0.5 * np.log(2 * np.pi)
    return rmse, nrmse, r2, maxae, mae, cicp, nmciw, nmciw_sem, nll


def plot_x_mean_2_theta(X_test_mean, theta_test, theta_test_approx, theta_test_approx_mean, filename=None):
    CI = confidence_interval(theta_test_approx)
    fig, ax = plt.subplots(2, 3, figsize=(12, 10))
    ax = ax.flat
    for j in range(len(param_names)):
        ax[j].scatter(X_test_mean, theta_test[:, j], color='black')
        ax[j].plot((X_test_mean, X_test_mean), (CI[0][:, j], CI[1][:, j]), color='#5c92e8', alpha=0.5)
        for i in range(n_test):
            if CI[0][i, j] <= theta_test[i, j] <= CI[1][i, j]:
                ax[j].scatter(X_test_mean[i], theta_test_approx_mean[i, j], color='green', alpha=0.5)
            else:
                ax[j].scatter(X_test_mean[i], theta_test_approx_mean[i, j], color='red', alpha=0.5)
        ax[j].set_xlabel(r'$\overline{x}$')
        ax[j].set_ylabel(param_names[j])
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close()


def plot_monte_carlo_dropout_inference(model, X_test, theta_test, n_dropout, figsize=(12, 10), filename=None):
    theta_test_approx_means = []
    theta_test_approx_vars = []
    CIs = [[], []]
    for i in range(n_dropout):
        model(torch.tensor(theta_test[0:1]), X_test[0:1])
        theta_test_approx = model.sample(X_test, n_samples)
        theta_test_approx = theta_test_approx.detach().cpu().numpy()
        ci = confidence_interval(theta_test_approx)
        CIs[0].append(np.expand_dims(ci[0], axis=0))
        CIs[1].append(np.expand_dims(ci[1], axis=0))
        theta_test_approx_means.append(np.mean(theta_test_approx, axis=0, keepdims=True))
        theta_test_approx_vars.append(np.var(theta_test_approx, axis=0, keepdims=True))
    theta_test_approx_means = np.concatenate(theta_test_approx_means, axis=0)
    theta_test_approx_vars = np.concatenate(theta_test_approx_vars, axis=0)
    CIs[0] = np.concatenate(CIs[0], axis=0)
    CIs[1] = np.concatenate(CIs[1], axis=0)
    epistemic_var = np.var(theta_test_approx_means, axis=0)
    aleatoric_var = np.mean(theta_test_approx_vars, axis=0)
    epistemic_var_mean = np.mean(epistemic_var, axis=0)
    aleatoric_var_mean = np.mean(aleatoric_var, axis=0)
    plt.rcParams['font.size'] = 12
    f, ax = plt.subplots(2, 3, figsize=figsize)
    ax = ax.flat

    for j in range(len(param_names)):
        for i in range(n_dropout):
            ax[j].scatter(theta_test[:, j], theta_test_approx_means[i, :, j], color='#5c92e8', s=40, alpha=0.8)
            ax[j].plot((theta_test[:, j], theta_test[:, j]), (CIs[0][i, :, j], CIs[1][i, :, j]), color='#5c92e8', alpha=0.5)
        lower_lim = min(ax[j].get_xlim()[0], ax[j].get_ylim()[0])
        upper_lim = max(ax[j].get_xlim()[1], ax[j].get_ylim()[1])
        ax[j].set_xlim((lower_lim, upper_lim))
        ax[j].set_ylim((lower_lim, upper_lim))
        ax[j].plot(ax[j].get_xlim(), ax[j].get_xlim(), '--', color='black')

        if j == 0:
            ax[j].set_xlabel('True')
            ax[j].set_ylabel('Estimated')
        ax[j].set_title(f'{param_names[j]}: '+r'$\sigma_e^2$'+'={:4f}, '.format(epistemic_var_mean[j])+"\n"+
                        r'$\sigma_a^2$'+'={:4f}'.format(aleatoric_var_mean[j]))
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
    # f.suptitle('uncertainty quantification')
    f.tight_layout()
    plt.show()
    if filename is not None:
        f.savefig(filename)
        plt.close()
    return epistemic_var_mean, aleatoric_var_mean


def monte_carlo_dropout_uncertainty(model, X_test, theta_test, n_dropout, n_samples):
    theta_test_approx_means = []
    theta_test_approx_vars = []
    CIs = [[], []]
    for i in range(n_dropout):
        model(torch.tensor(theta_test), X_test)
        theta_test_approx = model.sample(X_test, n_samples)
        theta_test_approx = theta_test_approx.detach().cpu().numpy()
        ci = confidence_interval(theta_test_approx)
        CIs[0].append(np.expand_dims(ci[0], axis=0))
        CIs[1].append(np.expand_dims(ci[1], axis=0))
        theta_test_approx_means.append(np.mean(theta_test_approx, axis=0, keepdims=True))
        theta_test_approx_vars.append(np.var(theta_test_approx, axis=0, keepdims=True))
    theta_test_approx_means = np.concatenate(theta_test_approx_means, axis=0)
    theta_test_approx_vars = np.concatenate(theta_test_approx_vars, axis=0)
    CIs[0] = np.concatenate(CIs[0], axis=0)
    CIs[1] = np.concatenate(CIs[1], axis=0)
    epistemic_var = np.var(theta_test_approx_means, axis=0)
    aleatoric_var = np.mean(theta_test_approx_vars, axis=0)
    epistemic_var_mean = np.mean(epistemic_var, axis=0)
    aleatoric_var_mean = np.mean(aleatoric_var, axis=0)
    return epistemic_var_mean, aleatoric_var_mean


if __name__ == '__main__':

    param_names = [r'$y_0$', r'$\mu$', r'$\l_f$']
    theta_dim = 3
    cond_dim = 5
    n_blocks = 4
    batch_size = 100
    train_size = 400
    epochs = 20000
    summary_dim = 64
    hidden_layer_size = 64
    conv_out = 64
    conv_kernel = 6
    ar_window = 0
    gru_out = 64
    skip_steps = [6]
    reccs_out = [64]
    iteration_per_epoch = train_size//batch_size
    l2_reg = 1e-5
    n_test = 200
    n_samples = 100
    n_latent = 20
    n_dropout = 20
    grad_clip = 1e+6
    dropout = True
    init_bias = 0.001
    kernel_gamma = 0.5
    max_explr_epoch = 2000
    start_lr = 1e-3
    noise_ratio = 0
    dropout_rate = 0.1

    data_path = 'data'
    vehicle = 3
    X_train, theta_train, X_val, theta_val, X_test, theta_test = \
        load_data(data_path + os.sep + f'vehicle_{vehicle}_param_STD_2000.npy',
                  data_path + os.sep + f'vehicle_{vehicle}_data_STD_2000.npy',
                  split0=train_size, split1=1600, split2=1800, split3=2000)
    model = ConditionalINN(theta_dim, cond_dim, summary_dim, conv_out, conv_kernel, gru_out, skip_steps, reccs_out,
                           ar_window, n_blocks, hidden_layer_size)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, theta_train),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, theta_val),
                                             batch_size=batch_size if theta_val.shape[0] > batch_size
                                             else theta_val.shape[0], shuffle=True)
    gamma = kernel_gamma ** (1 / max_explr_epoch)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = op.Adam(trainable_parameters, start_lr, weight_decay=l2_reg)
    lr_scheduler1 = op.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # lr_scheduler2 = torchcontrib.optim.SWA(optimizer, swa_freq=10, swa_lr=start_lr*kernel_gamma**3)
    torch.autograd.set_detect_anomaly(True)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(init_bias)

    model.apply(init_weights)
    train_loss = []
    val_loss = []
    early_stopping = EarlyStopping(patience=1000, verbose=True, delta=0.0001,
                                   path=f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5.pt')
    model.to(device)
    for ep in range(1, epochs + 1):

        with tqdm(total=epochs, desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:

            ml_loss, norm_loss, log_loss, Z, ml_val, norm_val, log_val = \
                train_model(model, optimizer, train_loader, val_loader, iteration_per_epoch, lr_scheduler1, ep)
            p_bar.set_postfix_str("ml_loss {0:.5f}, ml_val {1:.5f}".format(ml_loss, ml_val))
            p_bar.update(ep)
            train_loss.append(ml_loss)
            val_loss.append(ml_val)

        early_stopping(ml_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    np.save(
        f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5_train_loss.npy',
        np.array(train_loss))
    np.save(
        f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5_val_loss.npy',
        np.array(val_loss))

    # model.load(f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5.pt')
    # theta_approx = model.sample(X_test, n_samples, dropout=False).detach().numpy()
    # np.save(f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5_theta_approx.npy', theta_approx)
    # plot_inference_scatter_CI(theta_approx, theta_test, n_samples, param_names, figsize=(10, 4),
    #                           filename=f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5_inference.png')
    # plot_distribution_data_set(theta_approx, theta_test,
    # filename=f'results/BayesFlow_dropout_in_cINN_with_BN_LSTNet_vehicle_{vehicle}_STD_skip6_train_{train_size}_seq5_distribution.png')
    # bh_dists = [bh_dist(theta_approx.mean(axis=0)[:, i], theta_test.detach().numpy()[:, i]) for i in range(theta_dim)]
    # rmse, nrmse, r2, maxae, mae, cicp, nmciw, nmciw_sem, nll = metrics_summary(theta_test.detach().numpy(), theta_approx,