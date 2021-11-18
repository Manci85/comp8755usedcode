#!/usr/bin/env python
from __future__ import print_function
import time

import json
import os
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
from ptflops import get_model_complexity_info
# from sklearn.metrics import confusion_matrix

from notification.email_sender import send_email
from utils_dir.utils_io import mv_py_files_to_dir

from processor.args import get_parser

# Email notifications
from utils_dir.utils_math import dct_2_no_sum_parallel, gen_dct_on_the_fly
from utils_dir.utils_result import get_result_confusion_jsons
from utils_dir.utils_visual import azure_kinect_post_visualize

email_receivers = ['zhenyue.qin@anu.edu.au']

print('--- PyTorch version: ', torch.__version__, '---')
gpu_nums = torch.cuda.device_count()
print('--- Detected GPU #: ', gpu_nums, '---')
for a_gpu_id in range(gpu_nums):
    print('------ GPU {}'.format(a_gpu_id), torch.cuda.get_device_name(a_gpu_id), '---')
print('Executing main program ... ')

import numpy as np
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import apex

from utils import count_params, import_class, get_current_time

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Processor():
    """Processor for Skeleton-based Action Recognition"""

    def __init__(self, arg):
        self.arg = arg
        if arg.phase == 'train':
            # Added control through the command line

            # Debug mode args
            try:
                arg.train_feeder_args['debug'] = self.arg.debug or arg.train_feeder_args['debug']
                arg.test_feeder_args['debug'] = self.arg.debug or arg.test_feeder_args['debug']
            except KeyError:
                arg.train_feeder_args['debug'] = arg.test_feeder_args['debug'] = False

            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')
        elif arg.phase == 'model_features':
            logdir = os.path.join(arg.work_dir, 'analyze_logs')
            self.analyse_writer = SummaryWriter(logdir, 'analyze')

        # More model args
        # self.arg.model_args['ablation'] = self.arg.ablation
        # self.arg.model_args['to_use_transformer'] = self.arg.to_use_transformer
        # self.arg.model_args['to_use_temporal_transformer'] = self.arg.to_use_temporal_transformer
        # self.arg.model_args['tbatch'] = self.arg.tbatch
        # GCN multiscale layers
        # if self.arg.gcn_ms_layer > 0:
        #     self.arg.model_args['num_gcn_scales'] = self.arg.gcn_ms_layer

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        print('self batch size here: ', self.arg.batch_size)
        self.load_data()

        if not arg.resume:
            self.global_step = 0
            self.lr = self.arg.base_lr
            self.best_acc = 0
            self.best_top_5_acc = 0
            self.best_acc_epoch = 0
        else:
            ckpt_state = torch.load(self.arg.checkpoint, map_location='cpu')
            self.arg.start_epoch = ckpt_state['lr_scheduler_states']['last_epoch']
            self.lr = ckpt_state['lr_scheduler_states']['_last_lr']
            self.best_acc = 0
            self.best_top_5_acc = 0
            self.best_acc_epoch = 0

        # Save args
        self.save_arg()
        print('args: ', json.dumps(arg.__dict__, indent=4, sort_keys=True))

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

        # sending emails
        self.global_sending_count = 0
        self.global_sending_freq = 5

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        # Copy model file and main
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        tgt_model_dir = os.path.join(self.arg.work_dir, 'py_dir', 'model')
        mv_py_files_to_dir(os.path.join(*os.path.split(__file__)[:-1], 'model'), tgt_model_dir)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        print('output device: ', output_device)
        self.model = Model(**self.arg.model_args).cuda(output_device)

        # Count computational cost
        # macs, params = get_model_complexity_info(self.model,
        #                                          (3, 300, 25, 2), as_strings=True,
        #                                          print_per_layer_stat=False, verbose=False)
        # self.print_log('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # self.print_log('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # self.has_calculated_model_flops = True

        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))

                # Delete unexpected keys
                for a_weight_key in weights.copy().keys():
                    if a_weight_key not in state.keys():
                        weights.pop(a_weight_key, None)

                self.print_log('Cannot find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        if 'is_using_pretrained_transformer' in self.arg.model_args and \
                self.arg.model_args['is_using_pretrained_transformer']:
            print('Loading transformer optimizers. ')
            for name, params in self.model.named_parameters():
                if 'pretrained_transformer' in name:
                    # print('catch one')
                    if 'norm' in name:
                        self.param_groups['pretrained_transformer'].append(params)
                    elif 'head' in name:
                        self.param_groups['pretrained_transformer'].append(params)
                else:
                    self.param_groups['other'].append(params)

            self.optim_param_groups = {
                'other': {'params': self.param_groups['other']},
                'pretrained_transformer': {'params': self.param_groups['pretrained_transformer']}
            }

        else:
            for name, params in self.model.named_parameters():
                self.param_groups['other'].append(params)

            self.optim_param_groups = {
                'other': {'params': self.param_groups['other']}
            }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            # print('torch.load(self.arg.checkpoint): ', torch.load(self.arg.checkpoint, map_location={'cuda:2': 'cuda:0'}))
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint, map_location='cpu')['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=self.arg.lr_decay)
        milestones_cp = self.lr_scheduler.milestones  # to use new milestones
        lr_decay_cp = self.lr_scheduler.gamma
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint, map_location='cpu')['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')
            # comment the following line if to use the previous milestones
            self.lr_scheduler.milestones = milestones_cp
            # comment the following line if to use the previous lr decay
            self.lr_scheduler.gamma = lr_decay_cp

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'get_model_features':
            to_shuffle_train = False
        else:
            to_shuffle_train = True
        to_shuffle_train = True  # Ensure shuffling
        if self.arg.phase == 'train' or self.arg.phase == 'get_model_features':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=to_shuffle_train,
                num_workers=self.arg.num_worker,
                drop_last=to_shuffle_train,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_acc_epoch': self.best_acc_epoch
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def get_feature_combo_data(self, data):
        if 'ntu' in self.arg.train_feeder_args['data_path']:
            datatype = 'ntu'
        elif 'kinetics' in self.arg.train_feeder_args['data_path']:
            datatype = 'kinetics'
        else:
            return data

        if datatype == 'ntu':
            if self.arg.feature_combo == 'angular':
                data = data[:, 6:, :, :, :]
            elif self.arg.feature_combo == 'joint':
                data = data[:, :3, :, :, :]
            elif self.arg.feature_combo == 'bone':
                data = data[:, 3:6, :, :, :]
            elif self.arg.feature_combo == 'joint+angular':
                data = torch.cat((data[:, :3, :, :, :], data[:, 6:, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'bone+angular':
                data = torch.cat((data[:, 3:6, :, :, :], data[:, 6:, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'no_head':
                data = data[:, :15, :, :, :]
            elif self.arg.feature_combo == 'joint+bone':
                data = data[:, :6, :, :, :]
            elif self.arg.feature_combo == 'joint':
                data = data[:, :3, :, :, :]
            elif self.arg.feature_combo == 'bone':
                data = data[:, 3:6, :, :, :]
            elif self.arg.feature_combo == 'local_bone_angle':
                data = torch.cat((data[:, :6, :, :, :], data[:, 6:7, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'central_angle':
                data = torch.cat((data[:, :6, :, :, :], data[:, 7:9, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'hand_angle':
                data = torch.cat((data[:, :6, :, :, :], data[:, 9:11, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'arm_angle':
                data = torch.cat((data[:, :6, :, :, :], data[:, 11:13, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'leg_angle':
                data = torch.cat((data[:, :6, :, :, :], data[:, 13:15, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'local_center_figure_hand':
                data = data[:, :12, :, :, :]
            elif self.arg.feature_combo == 'normal':
                data = data[:, 6:9, :, :, :]
            elif self.arg.feature_combo == 'joint+local_center_figure_hand':
                data = torch.cat((data[:, :3, :, :, :], data[:, 6:12, :, :, :]), dim=1)

            # Analyze different angles
            ## Local angle
            elif self.arg.feature_combo == 'joint+local':
                data = torch.cat((data[:, :3, :, :, :], data[:, 6:7, :, :, :]), dim=1)
            ## Center angle
            elif self.arg.feature_combo == 'joint+unfixed_axis':
                data = torch.cat((data[:, :3, :, :, :], data[:, 7:8, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+fixed_axis':
                data = torch.cat((data[:, :3, :, :, :], data[:, 8:9, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+center':
                data = torch.cat((data[:, :3, :, :, :], data[:, 7:9, :, :, :]), dim=1)
            ## Finger angle
            elif self.arg.feature_combo == 'joint+left_finger':
                data = torch.cat((data[:, :3, :, :, :], data[:, 9:10, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+right_finger':
                data = torch.cat((data[:, :3, :, :, :], data[:, 10:11, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+finger':
                data = torch.cat((data[:, :3, :, :, :], data[:, 9:11, :, :, :]), dim=1)
            ## Part angle
            elif self.arg.feature_combo == 'joint+hand':
                data = torch.cat((data[:, :3, :, :, :], data[:, 11:12, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+elbow':
                data = torch.cat((data[:, :3, :, :, :], data[:, 12:13, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+knee':
                data = torch.cat((data[:, :3, :, :, :], data[:, 13:14, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+foot':
                data = torch.cat((data[:, :3, :, :, :], data[:, 14:15, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+part':
                data = torch.cat((data[:, :3, :, :, :], data[:, 11:15, :, :, :]), dim=1)
        elif datatype == 'kinetics':
            if self.arg.feature_combo == 'joint':
                data = torch.cat((data[:, :2, :, :, :], data[:, -1:, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'bone':
                data = torch.cat((data[:, 2:4, :, :, :], data[:, -1:, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'joint+angle':
                data = torch.cat((data[:, :2, :, :, :], data[:, 4:, :, :, :]), dim=1)
            elif self.arg.feature_combo == 'bone+angle':
                data = torch.cat((data[:, 2:4, :, :, :], data[:, 4:, :, :, :]), dim=1)

        # Fill the second empty person
        if False:
            for a_data_idx in range(len(data)):
                if torch.sum(torch.abs(data[a_data_idx][:, :, :, 1])) < 1e-3:
                    data[a_data_idx][:, :, :, 1] = data[a_data_idx][:, :, :, 0]

        return data

    def train(self, epoch, save_model=False, save_training_score=False):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step(epoch=epoch)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}, Total Batch: {len(loader)}')

        self.has_calculated_model_flops = False

        # process = tqdm(loader, dynamic_ncols=True)
        tr_score_batches = []
        for batch_idx, (data, label, index) in enumerate(loader):
            self.global_step += 1

            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

            # feature selection
            data = self.get_feature_combo_data(data)
            # self.print_log(f'{batch_idx}: Sum of data: {torch.sum(data)}')

            # for i in range(5):
            #     azure_kinect_post_visualize(data[i].unsqueeze(0).cpu().numpy(),
            #                                 f'test_fields/skeleton_label_test/label_{label[i]}_index_{batch_idx}_{i}.mp4')

            # 加入one hot
            if self.arg.to_add_onehot:
                one_hot = torch.eye(self.arg.model_args['num_point']).unsqueeze(1).unsqueeze(-1).unsqueeze(0).\
                    repeat(data.shape[0], 1, data.shape[2], 1, self.arg.model_args['num_person']).to(data.device)
                data = torch.cat((data, one_hot), dim=1)

            # Data loading time
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]

                # forward
                output = self.model(batch_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                # 记录training的out文件
                tr_score_batches.append(output.data.cpu().numpy())

                # 这并没有什么loss accumulation.
                # 但是注意, loss是在batch上平均了.
                # 就是每一个forward batch的loss变小,
                # 使得整个batch的loss大概相当于一个forward batch的原始loss.
                # 或许这样可以减小small batch的干扰程度.
                loss = self.loss(output, batch_label) / splits

                if self.arg.half:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                # process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())

                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # print training information
            if (batch_idx + 1) % self.arg.train_print_freq == 0:
                cur_mean_loss = np.mean(loss_values)
                self.print_log('Current batch: {} Current mean loss: {}'.format(
                    batch_idx + 1, cur_mean_loss))

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # training out scores
        tr_score = np.concatenate(tr_score_batches)
        tr_score_dict = dict(zip(loader.dataset.sample_name, tr_score))
        if save_training_score:
            with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, 'training'), 'wb') as f:
                pickle.dump(tr_score_dict, f)

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def get_model_features(self, loader_name=['test']):
        rtn_labels = []
        with torch.no_grad():
            self.model.to_fc_last = False
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            output_list = []
            for ln in loader_name:
                # process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                self.print_log('Get model outputs, '
                               'Total Batch: {}'.format(len(self.data_loader[ln])))
                for batch_idx, (data, label, index) in enumerate(self.data_loader[ln]):
                    rtn_labels.append(label)
                    data = data.float().cuda(self.output_device)

                    # feature selection
                    data = self.get_feature_combo_data(data)

                    # 加入one hot
                    if self.arg.to_add_onehot:
                        one_hot = torch.eye(self.arg.model_args['num_point']).unsqueeze(1).unsqueeze(-1).unsqueeze(0). \
                            repeat(data.shape[0], 1, data.shape[2], 1, self.arg.model_args['num_person']).to(
                            data.device)
                        data = torch.cat((data, one_hot), dim=1)

                    if batch_idx % 100 == 0:
                        print('Current Batch: ', batch_idx)
                    output = self.model(data, set_to_fc_last=False)
                    output_list.append(output.cpu())

        output_list = torch.cat(output_list, dim=0)
        rtn_labels = torch.cat(rtn_labels, dim=0)
        # Empty cache after evaluation
        empty_cache_device = self.arg.device[-1]
        with torch.cuda.device(empty_cache_device):
            torch.cuda.empty_cache()
        return output_list, rtn_labels

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.dct_K = 8
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                # process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                self.print_log('Eval Epoch: {}, Total Batch: {}'.format(epoch+1, len(self.data_loader[ln])))
                for batch_idx, (data, label, index) in enumerate(self.data_loader[ln]):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    # data_noise = torch.randn_like(data).to(data.device)
                    # data += 0.1 * data_noise  # 加噪音
                    # data += 0.4  # 平移
                    # data = gen_dct_on_the_fly(data, K=self.dct_K)  # DCT on the fly

                    # feature selection
                    data = self.get_feature_combo_data(data)

                    # 加入one hot
                    if self.arg.to_add_onehot:
                        one_hot = torch.eye(self.arg.model_args['num_point']).to(data.device).\
                            unsqueeze(1).unsqueeze(-1).unsqueeze(0). \
                            repeat(data.shape[0], 1, data.shape[2], 1, self.arg.model_args['num_person'])
                        data = torch.cat((data, one_hot), dim=1).to(data.device)

                    if batch_idx % 100 == 0:
                        print('Current Batch: ', batch_idx)
                    # print('data: ', data.shape)
                    # print('label: ', label.shape)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i].item()) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy >= self.best_acc:
                self.best_acc = accuracy
                self.best_top_5_acc = self.data_loader[ln].dataset.top_k(score, 5)
                self.best_acc_epoch = epoch + 1
                save_score = True

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')
            self.print_log(f'\tCurrent Best top-1 accuracy: {100 * self.best_acc:.2f}%')

            # get confusion matrix
            if self.best_acc_epoch == epoch + 1:
                if 'ntu' in self.arg.train_feeder_args['data_path'] or \
                        'test_feeding_data' in self.arg.train_feeder_args['data_path'] and 'ind' not in self.arg.train_feeder_args['data_path']:
                    acc_dir = os.path.join(self.arg.work_dir, 'accuracy_info')
                    if not os.path.exists(acc_dir):
                        os.makedirs(acc_dir)
                    acc_f_name_prefix = '{}/epoch_{}_{}'.format(acc_dir, epoch + 1, ln)
                    predicted_labels = score.argsort()[:, -1]
                    get_result_confusion_jsons(
                        self.data_loader[ln].dataset.label, predicted_labels,
                        self.arg.train_feeder_args['data_path'],
                        acc_f_name_prefix
                    )

            ts_email_msg = 'Incoming transmission ... <br>' \
                           'Current experiment progress: <br>' \
                           f'Model name: {self.arg.work_dir} <br>' \
                           f'Current Testing Epoch: {epoch} <br> ' \
                           f'\tTop {1}: {100 * self.data_loader[ln].dataset.top_k(score, 1):.2f}% <br> ' \
                           f'\tTop {5}: {100 * self.data_loader[ln].dataset.top_k(score, 5):.2f}%  <br> ' \
                           f'\tBest Top {1}: {100 * self.best_acc:.2f}% <br> '

            if self.global_sending_count % self.global_sending_freq == 0:
                send_email(email_receivers, 'test_end', ts_email_msg)
            self.global_sending_count += 1

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        empty_cache_device = self.arg.device[-1]
        with torch.cuda.device(empty_cache_device):
            torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            # self.print_log(self.model)
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1) % 5 == 0

                eval_model = (((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch)) and \
                    epoch + 1 >= self.arg.eval_start
                self.train(epoch, save_model=save_model, save_training_score=False)
                # print('eval model: ', self.arg.log_interval)
                if eval_model:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                if self.best_acc_epoch == epoch + 1:
                    self.save_weights(epoch=f'best_model_epoch_{epoch+1}')
                    # self.save_weights(epoch + 1)
                    self.save_checkpoint(epoch + 1)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best top-1 accuracy: {self.best_acc}')
            self.print_log(f'Best top-5 accuracy: {self.best_top_5_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

            email_msg = '报! 最终结果!  <br>' \
                        '程序运行已结束. 以下为最终结果: <br>' \
                        f'Model name: {self.arg.work_dir} <br>' \
                        f'Best top-1 accuracy: {self.best_acc} <br> ' \
                        f'Best top-5 accuracy: {self.best_top_5_acc} <br> ' \
                        f'Epoch number: {self.best_acc_epoch} <br> ' \
                        f'Weight decay: {self.arg.weight_decay} <br> ' \
                        f'Base LR: {self.arg.base_lr} <br> ' \
                        f'Batch Size: {self.arg.batch_size} <br> ' \
                        f'Forward Batch Size: {self.arg.forward_batch_size} <br> ' \
                        f'Test Batch Size: {self.arg.test_batch_size} <br><br> ' \
                        f'沃伦堡瞭望塔 持续为您观察. <br>'
            send_email(['zhenyue.qin@anu.edu.au'], 'exp_end', email_msg)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.arg.eval_start = -1
            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')

        elif self.arg.phase == 'get_model_features':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.arg.eval_start = -1
            model_features, model_labels = self.get_model_features(
                loader_name=['train']
            )
            model_labels = model_labels.cpu().numpy()
            self.print_log('model_features: {}'.format(model_features.shape))
            fea_save_name = os.path.join(self.arg.work_dir, 'model_features.npy')
            np.save(fea_save_name, model_features.numpy())
            self.analyse_writer.add_embedding(model_features, metadata=model_labels)
            self.print_log('Done.\n')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    p.num_worker = p.num_worker * 2

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    # if arg.phase == 'train':
    arg.work_dir = os.path.join(arg.work_dir, get_current_time())
    init_seed(arg.seed)
    print('Initial args: \n', arg)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()

    # try:
    #     main()
    # except Exception as e:
    #     pass
    #
    #     error_msg = f'{e}'
    #     send_email(['zhenyue.qin@anu.edu.au'],
    #                'error',
    #                error_msg)
    #     raise RuntimeError('Error caught during running. Report email has been sent. ')
