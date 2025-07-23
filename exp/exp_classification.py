import sys

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    # 加载模型
    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            # TODO: DP -> DDP
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据集
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # TODO: 默认优化器是RAdam，后续可以更换优化器
    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # TODO: Default loss
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # TODO: 这里用了argmax，下面的acc其实就是top1
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        # 在这里调用_get_data时设置了flag
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {:03d} cost time: {:.4f}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return


if __name__ == "__main__":
    print("\n------------ start ------------\n")

    # windows下注意要把num_workers设为0，因为windows/macos下，会使用spawn方式创建多线程，spawn创建子线程的方式就会用到pickle
    # 其会使用pickle对对象进行序列化，然后传给所有子线程
    # 而局部函数和匿名函数无法使用pickle序列化，全局函数或类实例可以

    import argparse


    def argparser():
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, required=False, default='classification',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        # 1为训练，0为验证/测试
        parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=False, default='EthanolConcentration', help='model id')
        parser.add_argument('--model', type=str, required=False, default='TimesNet',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str, required=False, default='UEA', help='dataset type')
        parser.add_argument('--root_path', type=str, default='G:/Time-Series-Library/dataset/JapaneseVowels',
                            help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=96,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='Exp', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                            default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False,
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true",
                            help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true",
                            help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true",
                            help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true",
                            help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true",
                            help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true",
                            help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

        # TimeXer
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')

        args = parser.parse_args()

        return args


    args = argparser()

    exp = Exp_Classification(args)  # set experiments
    train_data, train_loader = exp._get_data(flag='TRAIN')

    print("len(train_data): {}".format(len(train_data)))
    print("len(train_loader): {}".format(len(train_loader)))

    train_steps = len(train_loader)
    for i, (batch_x, label, padding_mask) in enumerate(train_loader):
        print("\n----- i: {} -----".format(i))
        print(batch_x.shape)
        print(batch_x)
        print(label.shape)
        print(padding_mask.shape)
        print(padding_mask)
        sys.exit(-1)
    print("\n------------- end -------------\n")
