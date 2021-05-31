import os, sys
import torch
import numpy as np
import argparse
from argparse import Namespace
import tqdm
import six
from scipy import stats
import pandas as pd
import pickle

from helper import  set_seeds
from torch.utils.data import DataLoader, TensorDataset
from datasets.datasets import get_scaled_data, get_synthetic_data
from utils.q_model_ens import QModelEns
from losses import batch_qr_loss, batch_interval_loss
import helper
sys.modules['sklearn.externals.six'] = six

np.warnings.filterwarnings('ignore')

os.environ["MKL_CBWR"] = 'AUTO'

results_path = helper.results_path

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_loss_fn(loss_name):

    if loss_name == 'batch_qr':
        fn = batch_qr_loss
    elif loss_name == 'batch_int':
        fn = batch_interval_loss
    else:
        raise ValueError('loss arg not valid')

    return fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    parser.add_argument('--seed_begin', type=int, default=None,
                        help='random seed')
    parser.add_argument('--seed_end', type=int, default=None,
                        help='random seed')

    parser.add_argument('--data', type=str, default='',
                        help='dataset to use')

    parser.add_argument('--num_q', type=int, default=30,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--nl', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--hs', type=int, default=64,
                        help='hidden size')

    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--loss', type=str,
                        help='specify type of loss')

    parser.add_argument('--corr_mult', type=float, default=0.,
                        help='correlation penalty multiplier')

    parser.add_argument('--hsic_mult', type=float, default=0.,
                        help='correlation penalty multiplier')

    parser.add_argument('--ds_type', type=str, default="",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.4,
                        help='ratio of test set size')

    parser.add_argument('--save_training_results', type=int, default=0,
                        help='1 for saving results during training, or 0 for not saving')

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device

    return args


def update_results_during_training(model, x, y, set_name, results_dict, alpha):
    with torch.no_grad():
        if len(x) == 0 or len(y) == 0:
            return
        y = y.reshape(-1).to(device)
        idx = np.random.permutation(len(x))  # [:len(xx)]
        x = x[idx].to(device)
        quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2]).to(device)
        test_preds = model.predict_q(
            x, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None, use_best_va_model=False
        )
        test_preds.detach().cpu().numpy()
        y_upper = test_preds[:, 1].detach().cpu().numpy()
        y_lower = test_preds[:, 0].detach().cpu().numpy()

        if torch.is_tensor(y):
            curr_y = y.cpu().detach().numpy()[idx]
        else:
            curr_y = y[idx]
        in_the_range = ((curr_y >= y_lower) & (curr_y <= y_upper))
        lengths = (y_upper - y_lower)

        if 'pearsons_correlation' + '_over_' + set_name not in results_dict:
            results_dict['pearsons_correlation' + '_over_' + set_name] = []

        results_dict['pearsons_correlation' + '_over_' + set_name] += [
            stats.pearsonr(in_the_range, lengths)[0]]

        if 'coverage' + '_over_' + set_name not in results_dict:
            results_dict['coverage' + '_over_' + set_name] = []

        results_dict['coverage' + '_over_' + set_name] += [np.mean(in_the_range)]

        if 'interval_lengths' + '_over_' + set_name not in results_dict:
            results_dict['interval_lengths' + '_over_' + set_name] = []

        results_dict['interval_lengths' + '_over_' + set_name] += [np.mean(lengths)]


if __name__ == '__main__':

    args = parse_args()
    args.num_ens = 1
    args.boot = 0

    POSSIBLE_REAL_DATA_NAMES = ['kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21', 'facebook_1', 'facebook_2', 'blog_data', 'bio', 'scaled_bio', 'bike']

    REAL_DATA = 'real data'
    SYN_DATA = 'synthetic data'
    if 'syn' in args.ds_type.lower():
        data_type = SYN_DATA
    elif 'real' in args.ds_type.lower():
        data_type = REAL_DATA
    else:
        raise RuntimeError('Must decide dataset type!')

    if data_type == REAL_DATA:

        if args.data != '' and args.data.replace("scaled_", '') not in POSSIBLE_REAL_DATA_NAMES:
            raise RuntimeError('Must choose possible data name!')

        if args.data.replace("scaled_", '') in POSSIBLE_REAL_DATA_NAMES:
            REAL_DATA_NAMES = [args.data]
        else:
            REAL_DATA_NAMES = []
        DATA_NAMES = REAL_DATA_NAMES
    else:
        assert data_type == SYN_DATA
        if str(args.data) in [str(i) for i in range(0, 10)]:
            SYN_DATA_NAMES = [args.data]
        else:
            SYN_DATA_NAMES = ['3', '10']
        DATA_NAMES = SYN_DATA_NAMES


    if args.seed is not None:
        SEEDS = [args.seed]
    elif args.seed_begin is not None and args.seed_end is not None:
        SEEDS = range(args.seed_begin, args.seed_end)
    else:
        SEEDS = range(0, 30)

    save_results_during_training = bool(args.save_training_results)

    print('DEVICE: {}'.format(args.device))

    alpha = 0.1

    args_summary = str("loss=" + args.loss + "_bs=" + str(args.bs) + "_corr_mult=" +
                       str(args.corr_mult) +'_hsic_mult='+str(args.hsic_mult))

    if 'int' in args.loss:
        TRAINING_OVER_ALL_QUANTILES = True
    elif 'qr' in args.loss:
        TRAINING_OVER_ALL_QUANTILES = False
    else:
        assert False

    for d in DATA_NAMES:
        if save_results_during_training:
            results_during_training = {}
            for s in SEEDS:
                results_during_training[s] = {}

        for s in tqdm.tqdm(SEEDS):

            args.data = str(d)
            args.seed = s


            set_seeds(args.seed)

            if data_type == REAL_DATA:
                # Fetching data
                data_args = Namespace(dataset=args.data, seed=args.seed)
                data_out = get_scaled_data(args)

            else:
                minority_group_uncertainty = d
                data_out = get_synthetic_data(args, minority_group_uncertainty)

                unscaled_x_train = data_out.unscaled_x_train
                unscaled_x_test = data_out.unscaled_x_test
                unscaled_x_tr = data_out.unscaled_x_tr
                unscaled_x_va = data_out.unscaled_x_va
                n_groups = data_out.n_groups
                group_feature = data_out.group_feature
                syn_x_train = data_out.syn_x_train  # train + validation x
                syn_y_train = data_out.syn_y_train  # train + validation y

            x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = \
                data_out.x_tr, data_out.x_va, data_out.x_te, data_out.y_tr, \
                data_out.y_va, data_out.y_te, data_out.y_al
            y_range = (y_al.max() - y_al.min()).item()

            # Fetching data
            data_args = Namespace(dataset=args.data, seed=args.seed)

            # creating the model
            num_tr = x_tr.shape[0]
            dim_x = x_tr.shape[1]
            dim_y = y_tr.shape[1]

            model_ens = QModelEns(input_size=dim_x+1, output_size=dim_y,
                                  hidden_size=args.hs, num_layers=args.nl, dropout=args.dropout,
                                  lr=args.lr, wd=args.wd,
                                  num_ens=args.num_ens, device=args.device)

            # Data loader
            loader = DataLoader(TensorDataset(x_tr, y_tr),
                                shuffle=True,
                                batch_size=args.bs)

            # Loss function
            loss_fn = get_loss_fn(args.loss)
            batch_loss = True if 'batch' in args.loss else False

            """ train loop """
            tr_loss_list = []
            va_loss_list = []
            te_loss_list = []

            for ep in range(args.num_ep):

                if model_ens.done_training:
                    print('Done training ens at EP {}'.format(ep))
                    break

                # Take train step
                ep_train_loss = []  # list of losses from each batch, for one epoch
                for (xi, yi) in loader:
                    xi, yi = xi.to(args.device), yi.to(args.device)
                    if TRAINING_OVER_ALL_QUANTILES:
                        q_list = torch.rand(args.num_q)
                    else:
                        q_list = torch.Tensor([alpha / 2])
                    loss = model_ens.loss(loss_fn, xi, yi, q_list,
                                          batch_q=batch_loss,
                                          take_step=True, args=args)
                    ep_train_loss.append(loss)

                ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
                tr_loss_list.append(ep_tr_loss)

                # Validation loss
                x_va, y_va = x_va.to(args.device), y_va.to(args.device)
                if TRAINING_OVER_ALL_QUANTILES:
                    va_te_q_list = torch.linspace(0.01, 0.99, 99)
                else:
                    va_te_q_list = torch.Tensor([alpha/2, 1-alpha/2])
                ep_va_loss = model_ens.update_va_loss(
                    loss_fn, x_va, y_va, va_te_q_list,
                    batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
                    args=args
                )
                va_loss_list.append(ep_va_loss)

                if save_results_during_training:
                    if data_type == SYN_DATA:
                        for (unscaled_x, x,y,set_name) in [(unscaled_x_tr, x_tr, y_tr, 'train'),
                                                           (unscaled_x_va, x_va, y_va, 'validation'), (unscaled_x_test,x_te, y_te, 'test')]:
                            update_results_during_training(model_ens, x, y, set_name,
                                                           results_during_training[s], alpha)
                            for group_number in range(n_groups):
                                group_idx = (unscaled_x[:, group_feature] == group_number)
                                update_results_during_training(model_ens, x[group_idx], y[group_idx],
                                                               set_name+'_group_'+str(group_number),
                                                               results_during_training[s], alpha)

                    else:
                        for (x,y,set_name) in [(x_tr, y_tr, 'train'), (x_va, y_va, 'validation'), (x_te, y_te, 'test')]:
                            update_results_during_training(model_ens, x, y, set_name,
                                                           results_during_training[s], alpha)



                # Printing some losses
                if (ep % 100 == 0) or (ep == args.num_ep-1):
                    # pass
                    print('EP:{}'.format(ep))
                    # print('Train loss {}'.format(ep_tr_loss))
                    # print('Val loss {}'.format(ep_va_loss))
                    # print('Test loss {}'.format(ep_te_loss))

            # Move everything to cpu
            x_tr, y_tr, x_va, y_va, x_te, y_te = \
                x_tr.cpu(), y_tr.cpu(), x_va.cpu(), y_va.cpu(), x_te.cpu(), y_te.cpu()
            model_ens.use_device(torch.device('cpu'))


            if data_type == REAL_DATA:
                x_train = torch.cat((x_tr, x_va))
                y_train = torch.cat((y_tr, y_va))
            else:
                x_train = syn_x_train  # train + validation x
                y_train = syn_y_train  # train + validation y

            x_test = x_te
            y_test = y_te

            quantiles = torch.Tensor([alpha/2, 1-alpha/2])
            test_preds = model_ens.predict_q(
                x_te, quantiles, ens_pred_type='conf',
                recal_model=None, recal_type=None
            )
            test_preds.detach().cpu().numpy()

            train_preds = model_ens.predict_q(
                x_train, quantiles, ens_pred_type='conf',
                recal_model=None, recal_type=None
            )
            train_preds.detach().cpu().numpy()

            train_y_upper = train_preds[:, 1]
            train_y_lower = train_preds[:, 0]
            y_train = y_train.reshape(len(y_train))
            y_upper = test_preds[:, 1]
            y_lower = test_preds[:, 0]
            y_test = y_test.reshape(len(y_test))


            if data_type==SYN_DATA:
                synthetic_data_params={
                    'consider_groups': True,
                    'group_feature':group_feature
                }

                x_train = unscaled_x_train
                x_test = unscaled_x_test
            else:
                synthetic_data_params = {
                    'consider_groups': False
                }



            return_values = helper.calculate_results(x_train,
                                                               y_train,
                                                               x_test,
                                                               y_test,
                                                               y_upper,
                                                               y_lower,
                                                               train_y_upper,
                                                               train_y_lower,
                                                               **synthetic_data_params)

            if data_type == SYN_DATA:
                curr_results_path = results_path + 'syn_data/minority_group_uncertainty=' + str(minority_group_uncertainty) + '/'
            else:
                dataset_number = ''
                curr_results_path = results_path + 'real_data/' + d + '/'

            curr_results_path += args_summary

            helper.create_folder_if_it_doesnt_exist(curr_results_path)
            file_name = curr_results_path + '/seed=' + str(s)

            pd.DataFrame(return_values, index=[s]).to_csv(file_name + '.csv')

            if save_results_during_training:
                data_name = d
                during_training_results_saving_dir = f'./results/during_training/{data_type.replace(" ", "_")}/{data_name}/'
                helper.create_folder_if_it_doesnt_exist(during_training_results_saving_dir)
                result_during_training_name = during_training_results_saving_dir + "results_during_training"

                if os.path.exists(result_during_training_name):
                    total_results_during_training = pickle.load(open(result_during_training_name, "rb"))
                else:
                    total_results_during_training = {}

                if args_summary not in total_results_during_training:
                    total_results_during_training[args_summary] = {}
                total_results_during_training[args_summary] = results_during_training

                pickle.dump(total_results_during_training, open(during_training_results_saving_dir+'results_during_training', "wb"))
            print()

