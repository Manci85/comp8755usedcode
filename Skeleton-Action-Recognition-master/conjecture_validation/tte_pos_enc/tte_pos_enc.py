import torch
import torch.nn as nn
import torch.optim as optim
import math

from utils_dir.utils_visual import plot_multiple_lines


def enc_sequence(a_seq, K, enc_type='dot'):
    N = len(a_seq)
    freq_bands = []

    basis_list = []
    for k in range(K+1):
        a_basis_list = []
        for n in range(0, N):
            a_basis_list.append(math.cos(math.pi / N * (n + 0.5) * k))
        basis_list.append(a_basis_list)
    basis_list = torch.tensor(basis_list).to('cuda')
    if enc_type == 'dot':
        a_seq_repeat = a_seq.transpose(1, 0).repeat(K+1, 1) * basis_list
    elif enc_type == 'tri_only':
        a_seq_repeat = basis_list
    elif enc_type == 'seq_only':
        a_seq_repeat = a_seq.transpose(1, 0).repeat(K+1, 1)
    else:
        raise NotImplementedError('Unsupported enc type. ')
    plot_multiple_lines(lines=basis_list.cpu().numpy())
    a_seq_repeat = a_seq_repeat.transpose(1, 0)
    return a_seq_repeat


class EncPosInfoChecker:
    def __init__(self):
        self.epoch_num = 1000
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.K = 8
        self.seq_len = 300
        self.enc_type = 'tri_only'

        self.build_frm_idx_net(self.K+1)
        self.train()

    def build_frm_idx_net(self, in_dim):
        self.idx_classifier = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).cuda()

    def gen_sequence(self, seq_len):
        self.a_seq = torch.randn((seq_len, 1)).cuda()
        self.a_seq = enc_sequence(self.a_seq, self.K, enc_type=self.enc_type)
        print('a seq: ', self.a_seq.shape)
        self.gt_idx = torch.range(1, seq_len).unsqueeze(-1).cuda() / 10

    def load_optimizer(self):
        self.optimizer = optim.SGD(
            self.idx_classifier.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay)
        self.loss_func = nn.L1Loss().cuda()

    def train(self):
        self.gen_sequence(self.seq_len)
        self.load_optimizer()
        for a_epoch in range(1, self.epoch_num+1):
            out = self.idx_classifier(self.a_seq)
            self.optimizer.zero_grad()
            loss = self.loss_func(out, self.gt_idx)
            loss.backward()
            self.optimizer.step()
            if a_epoch % 20 == 0:
                print('Current epoch: ', a_epoch, 'loss: ', loss.item())

        plot_multiple_lines([self.gt_idx.squeeze().cpu().numpy(),
                             out.detach().squeeze().cpu().numpy()],
                            labels=['GT', 'Out'])

if __name__ == '__main__':
    enc_pos_info_checker = EncPosInfoChecker()
    enc_pos_info_checker.train()
