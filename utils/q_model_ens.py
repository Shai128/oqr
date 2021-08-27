"""
Code copied from https://github.com/YoungseogChung/calibrated-quantile-uq
"""
import os, sys
from copy import deepcopy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch
from scipy.stats import norm as norm_distr
from scipy.stats import t as t_distr
from scipy.interpolate import interp1d
from torch import nn

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from NNKit.models.model import vanilla_nn

"""
Define wrapper uq_model class
All uq models will import this class
"""
class uq_model(object):

    def predict(self):
        raise NotImplementedError('Abstract Method')


""" QModelEns Utils """
def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def get_ens_pred_interp(unc_preds, taus, fidelity=10000):
    """
    unc_preds 3D ndarray (ens_size, 99, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, fidelity)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(xs, ys,
                            kind='linear',
                            fill_value=([0], [1]),
                            bounds_error=False)
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95, score_distr='z'):
    """
    unc_preds 3D ndarray (ens_size, num_tau, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    num_ens, num_tau, num_x = unc_preds.shape
    len_tau = taus.size

    mean_pred = np.mean(unc_preds, axis=0)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred / np.sqrt(num_ens)
    alpha = (1 - conf_level) # is (1-C)

    # determine coefficient
    if score_distr == 'z':
        crit_value = norm_distr.ppf(1- (0.5 * alpha))
    elif score_distr == 't':
        crit_value = t_distr.ppf(q=1- (0.5 * alpha), df=(num_ens-1))
    else:
        raise ValueError('score_distr must be one of z or t')

    gt_med = (taus > 0.5).reshape(-1, num_x)
    lt_med = ~gt_med
    assert gt_med.shape == mean_pred.shape == stderr_pred.shape
    out = (lt_med * (mean_pred - (float(crit_value) * stderr_pred)) +
           gt_med * (mean_pred + (float(crit_value) * stderr_pred))).T
    out = torch.from_numpy(out)
    return out


class QModelEns(uq_model):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, lr, wd,
                 num_ens, device):

        self.num_ens = num_ens
        self.device = device
        self.dropout = dropout
        self.model = [vanilla_nn(input_size=input_size, output_size=output_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 dropout=dropout).to(device)
                      for _ in range(num_ens)]
        self.optimizers = [torch.optim.Adam(x.parameters(),
                                            lr=lr, weight_decay=wd)
                           for x in self.model]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]
        self.done_training = False

    def use_device(self, device):
        self.device = device
        for idx in range(len(self.best_va_model)):
            self.best_va_model[idx] = self.best_va_model[idx].to(device)

        if device.type == 'cuda':
            for idx in range(len(self.best_va_model)):
                assert next(self.best_va_model[idx].parameters()).is_cuda

    def print_device(self):
        device_list = []
        for idx in range(len(self.best_va_model)):
            if next(self.best_va_model[idx].parameters()).is_cuda:
                device_list.append('cuda')
            else:
                device_list.append('cpu')
        print(device_list)

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args, weights=None):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y, x, q_list, self.device, args, weights=weights)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx], y, x,
                                             q_list, self.device, args)
                ens_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def loss_boot(self, loss_fn, x_list, y_list, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y_list[idx], x_list[idx], 
                                   q_list, self.device, args)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx], 
                               y_list[idx], x_list[idx], q_list, 
                               self.device, args)
                ens_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def update_va_loss(self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args, weights=None):
        with torch.no_grad():
            va_loss = self.loss(loss_fn, x, y, q_list, batch_q, take_step=False, args=args, weights=weights)

        # if torch.isnan(va_loss):
        #     print("va loss is nan!")

        for idx in range(self.num_ens):
            if self.keep_training[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.model[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        self.keep_training[idx] = False

        if not any(self.keep_training):
            self.done_training = True

        return va_loss


    #####
    def predict(self, cdf_in, conf_level=0.95, score_distr='z',
                 recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Only pass in cdf_in into model and return output
        If self is an ensemble, return a conservative output based on conf_bound
        specified by conf_level

        :param cdf_in: tensor [x, p], of size (num_x, dim_x + 1)
        :param conf_level: confidence level for ensemble prediction
        :param score_distr: 'z' or 't' for confidence bound coefficient
        :param recal_model:
        :param recal_type:
        :return:
        """

        if self.num_ens == 1:
            with torch.no_grad():
                if use_best_va_model:
                    pred = self.best_va_model[0](cdf_in)
                else:
                    pred = self.model[0](cdf_in)
        if self.num_ens > 1:
            pred_list = []
            if use_best_va_model:
                models = self.best_va_model
            else:
                models = self.model

            for m in models:
                with torch.no_grad():
                    pred_list.append(m(cdf_in).T.unsqueeze(0))

            unc_preds = torch.cat(pred_list, dim=0).detach().cpu().numpy()  # shape (num_ens, num_x, 1)
            taus = cdf_in[:, -1].flatten().cpu().numpy()
            pred = get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95,
                                           score_distr='z')
            pred = pred.to(cdf_in.device)

        return pred
    #####

    def predict_q(self, x, q_list=None, ens_pred_type='conf',
                recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """

        if q_list is None:
            q_list = torch.arange(0.01, 0.99, 0.01)
        else:
            q_list = q_list.flatten()

        if self.num_ens > 1:
            # choose function to make ens predictions
            if ens_pred_type == 'conf':
                ens_pred_fn = get_ens_pred_conf_bound
            elif ens_pred_type == 'interp':
                ens_pred_fn = get_ens_pred_interp
            else:
                raise ValueError('ens_pred_type must be one of conf or interp')

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []
        for p in q_list:
            if recal_model is not None:
                if recal_type == 'torch':
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == 'sklearn':
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError('recal_type incorrect')
            else:
                in_p = float(p)
            p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1).to(x.device)

            cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            cdf_pred = self.predict(cdf_in, use_best_va_model=use_best_va_model)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred)

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q)
        assert pred_mat.shape == (num_x, num_q)
        return pred_mat

        # ###
        # cdf_preds = []
        # for p in q_list:
        #     if recal_model is not None:
        #         if recal_type == 'torch':
        #             recal_model.cpu() # keep recal model on cpu
        #             with torch.no_grad():
        #                 in_p = recal_model(p.reshape(1, -1)).item()
        #         elif recal_type == 'sklearn':
        #             in_p = float(recal_model.predict(p.flatten()))
        #         else:
        #             raise ValueError('recal_type incorrect')
        #     else:
        #         in_p = float(p)
        #     p_tensor = (in_p * torch.ones(num_pts)).reshape(-1, 1)
        #     cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
        #
        #     ens_preds_p = []
        #     with torch.no_grad():
        #         for m in self.best_va_model:
        #             cdf_pred = m(cdf_in).reshape(num_pts, -1)
        #             ens_preds_p.append(cdf_pred.flatten())
        #
        #     cdf_preds.append(torch.stack(ens_preds_p, dim=0).unsqueeze(1))
        # ens_pred_mat = torch.cat(cdf_preds, dim=1).numpy()
        #
        # if self.num_ens > 1:
        #     assert ens_pred_mat.shape == (self.num_ens, num_q, num_pts)
        #     ens_pred = ens_pred_fn(ens_pred_mat, taus=q_list)
        # else:
        #     ens_pred = ens_pred_mat.reshape(num_q, num_pts)
        # return ens_pred
        # ###



class MSEModel(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim=64,
                 dropout=0.1,
                 device='cpu'):

        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.base_model = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        return self.base_model(x).squeeze()

    def epoch_loss(self, x_tr, y_tr, batch_size=64):
        # self.train()
        shuffle_idx = np.random.permutation(len(x_tr))
        x_train = x_tr[shuffle_idx]
        y_train = y_tr[shuffle_idx]
        epoch_loss = []
        for idx in range(0, x_train.shape[0], batch_size):
            self.optimizer.zero_grad()
            batch_x = x_train[idx: min(idx + batch_size, x_train.shape[0]), :]
            batch_y = y_train[idx: min(idx + batch_size, y_train.shape[0])]
            preds = self.forward(batch_x)
            loss = ((preds - batch_y) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            epoch_loss += [loss.cpu().item()]
        return np.mean(epoch_loss)

    def fit(self, x_tr, y_tr, x_val, y_val, batch_size=64, n_epochs=1000, wait=100):
        y_tr = y_tr.squeeze()
        y_val = y_val.squeeze()
        best_model = None
        best_val_loss = np.inf
        best_epoch = 0
        val_losses = []
        train_losses = []
        for e in range(n_epochs):
            loss = self.epoch_loss(x_tr, y_tr, batch_size=batch_size)
            train_losses += [loss]
            self.eval()
            with torch.no_grad():
                val_loss = ((self.forward(x_val) - y_val) ** 2).mean()
            val_losses += [val_loss.cpu().item()]
            if val_loss < best_val_loss:
                best_model = deepcopy(self.base_model)
                best_val_loss = val_loss
                best_epoch = e
            else:
                if e - best_epoch > wait:
                    break

        # plt.semilogy(val_losses, label="validation loss")
        # plt.semilogy(train_losses, label="train loss")
        # plt.xlabel("epoch")
        # plt.legend()
        # plt.show()
        self.base_model = best_model
        self.eval()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False






if __name__=='__main__':
    temp_model = QModelEns(input_size=1, output_size=1, hidden_size=10,
                           num_layers=2, lr=0.01, wd=0.0, num_ens=5,
                           device=torch.device('cuda:0'))
