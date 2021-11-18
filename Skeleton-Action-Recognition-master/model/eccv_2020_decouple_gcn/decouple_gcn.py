import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .dropSke import DropBlock_Ske
from .dropT import DropBlockT_1d

joint_num = 25

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske()
        self.dropT = DropBlockT_1d(block_size=41)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(
            torch.tensor(np.reshape(A.astype(np.float32), [3, 1, joint_num, joint_num]), dtype=torch.float32,
                         requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(
            torch.zeros(in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(
            torch.zeros(1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(joint_num))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(eye_array), requires_grad=False, device='cuda'),
                                 requires_grad=False)  # [c,joint_num,joint_num]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, joint_num, joint_num)
        D_list = torch.sum(A, 1).view(c, 1, joint_num)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat(
            [self.norm(learn_A[0:1, ...]), self.norm(learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum('nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(
            torch.tensor(np.sum(np.reshape(A.astype(np.float32), [3, joint_num, joint_num]), axis=0), dtype=torch.float32,
                         requires_grad=False, device='cuda'), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske()
        self.dropT_skip = DropBlockT_1d(block_size=41)

    def forward(self, x, keep_prob):
        x = self.tcn1(self.gcn1(x), keep_prob, self.A) + self.dropT_skip(
            self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=joint_num, num_person=2, groups=8, graph=None, graph_args=dict(),
                 in_channels=3, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, groups, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, groups)
        self.l3 = TCN_GCN_unit(64, 64, A, groups)
        self.l4 = TCN_GCN_unit(64, 64, A, groups)
        self.l5 = TCN_GCN_unit(64, 128, A, groups, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A, groups)
        self.l7 = TCN_GCN_unit(128, 128, A, groups)
        self.l8 = TCN_GCN_unit(128, 256, A, groups, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, groups)
        self.l10 = TCN_GCN_unit(256, 256, A, groups)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        # Rank pooling loss
        self.is_use_rank_pool = 'is_use_rank_pool' in kwargs and kwargs['is_use_rank_pool']
        # self.is_use_rank_pool = False
        if self.is_use_rank_pool:
            self.rank_pool_layer = nn.Linear(256, 1)

        if 'is_fgr_bly' in kwargs and kwargs['is_fgr_bly']:
            self.is_fgr_bly = True
            # 这个是bullying fine-grain用的fc
            self.bly_fgr_fc = nn.Linear(256, 1)
        else:
            self.is_fgr_bly = False

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)

        c_new = x.size(1)  # N*M,C,T,V
        t_dim = x.shape[2]
        x = x.view(N, M, c_new, t_dim, -1)
        x = x.permute(0, 1, 3, 4, 2)  # N, M, T, V, C
        x = x.mean(3)  # Global Average Pooling (Spatial)

        # Rank pooling
        if self.is_use_rank_pool:
            rank_pool_out = self.rank_pool_layer(x).squeeze()
            rank_pool_out = rank_pool_out.mean(1)

        x = x.mean(2)  # Global Average Pooling (Temporal)
        x = x.mean(1)  # Average pool number of bodies in the sequence

        # 这个是为了判断bullying的主动和被动
        if self.is_fgr_bly:
            bly_fgr_out = self.bly_fgr_fc(x).squeeze()

        if self.is_fgr_bly:
            return self.fc(x), bly_fgr_out
        else:
            other_outs = {}
            if self.is_use_rank_pool:
                other_outs['rank_pool'] = rank_pool_out

            return x, other_outs
