import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import TensorDataset
import torch
import math

ntu_bone_adj = {
    25: 12,
    24: 12,
    12: 11,
    11: 10,
    10: 9,
    9: 21,
    21: 21,
    5: 21,
    6: 5,
    7: 6,
    8: 7,
    22: 8,
    23: 8,
    3: 21,
    4: 3,
    2: 21,
    1: 2,
    17: 1,
    18: 17,
    19: 18,
    20: 19,
    13: 1,
    14: 13,
    15: 14,
    16: 15
}

ntu_bone_angle_pairs = {
    25: (24, 12),
    24: (25, 12),
    12: (24, 25),
    11: (12, 10),
    10: (11, 9),
    9: (10, 21),
    21: (9, 5),
    5: (21, 6),
    6: (5, 7),
    7: (6, 8),
    8: (23, 22),
    22: (8, 23),
    23: (8, 22),
    3: (4, 21),
    4: (4, 4),
    2: (21, 1),
    1: (17, 13),
    17: (18, 1),
    18: (19, 17),
    19: (20, 18),
    20: (20, 20),
    13: (1, 14),
    14: (13, 15),
    15: (14, 16),
    16: (16, 16)
}


class DataEncodingMachine:
    def __init__(self, feature_type, feature_source, data_status):
        # self.data_type = 'anubis/frt-bck'  # ANUBIS
        self.data_type = 'ntu120/xsub'  # NTU
        self.is_with_orig = True
        self.feature_source = feature_source  # joint OR bone
        self.data_status = data_status  # train OR val
        self.bch_sz = 1000

        # self.root_path = '/media/zhenyue-qin/local/' \
        #                  'Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
        # self.root_path = '/mnt/80F484E6F484E030/' \
        #                  'Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
        self.root_path = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                            'MS-G3D/data'

        self.trn_data_path = '{}/{}/train_data_{}.npy'.format(self.root_path, self.data_type,
                                                              self.feature_source)
        self.val_data_path = '{}/{}/val_data_{}.npy'.format(self.root_path, self.data_type,
                                                            self.feature_source)
        self.use_mmap = True

        if self.data_status == 'train':
            self.trn_data = self.load_data(self.trn_data_path)
            print('train data: ', self.trn_data.shape)
        elif self.data_status == 'val':
            self.val_data = self.load_data(self.val_data_path)
            print('val data: ', self.val_data.shape)

        if self.data_status == 'train':
            _, self.C, self.T, self.V, self.M = self.trn_data.shape
        elif self.data_status == 'val':
            _, self.C, self.T, self.V, self.M = self.val_data.shape
        self.K = 8  # 频率分量

        # if self.data_status == 'train':
        #     self.trn_dataloader = self.get_dataloader(self.trn_data, self.bch_sz)
        # elif self.data_status == 'val':
        #     self.val_dataloader = self.get_dataloader(self.val_data, self.bch_sz)

        ### Random Begin ###
        # self.feature_type = 'rand_0_1_dot'
        ### Random END ###

        ### Trigonometric Begin ###
        self.feature_type = feature_type
        self.inc_func = 'linear'
        self.periodic_fns = [torch.cos]
        if 'trig' in self.feature_type:
            self.prepare_period_fns()
        ### Trigonometric End ###

    @staticmethod
    def load_data(data_path, use_mmap=True):
        if use_mmap:
            data = np.load(data_path, mmap_mode='r')
        else:
            data = np.load(data_path)
        return data

    @staticmethod
    def get_dataloader(a_data, bch_sz):
        a_dataset = TensorDataset(torch.tensor(a_data))
        a_dataloader = torch.utils.data.DataLoader(
            dataset=a_dataset,
            batch_size=bch_sz,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        return a_dataloader

    @staticmethod
    def dct_2_parallel(bch_seq, K0=0, K1=None):
        bch_seq_rsp = bch_seq.view(-1, bch_seq.shape[1])
        N = bch_seq_rsp.shape[1]
        if K1 is None:
            K1 = N
        basis_list = []
        for k in range(K0, K1):
            a_basis_list = []
            for i in range(N):
                a_basis_list.append(math.cos(math.pi / N * (i + 0.5) * k))
            basis_list.append(a_basis_list)
        basis_list = torch.tensor(basis_list).to('cuda')
        dot_prod = torch.einsum('ab,bc->ac', bch_seq_rsp, basis_list.transpose(1, 0))
        return dot_prod.view(-1, K1 - K0)

    @staticmethod
    def idct_2_parallel(bch_seq, K=None):
        # N是分量个数, K是序列的长度
        bch_seq_rsp = bch_seq.view(-1, bch_seq.shape[1])
        N = bch_seq_rsp.shape[1]
        if K is None:
            K = N
        basis_list = []
        for k in range(K):
            a_basis_list = [0]
            for i in range(1, N):
                a_basis_list.append(math.cos(math.pi / N * (k + 0.5) * i))
            basis_list.append(a_basis_list)
        basis_list = torch.tensor(basis_list).to('cuda')
        dot_prod = torch.einsum('ab,bc->ac', bch_seq_rsp, basis_list.transpose(1, 0))
        bch_x0 = bch_seq_rsp[:, 0].unsqueeze(1).repeat(1, K) * 0.5
        unnorm_rst = dot_prod + bch_x0  # 弥补之前的第一位是0
        norm_rst = unnorm_rst * (2.0 / K)
        return norm_rst.view(-1, K)

    def discrete_cosine_transform_data(self, a_bch):
        a_bch_shape = a_bch.shape
        a_bch_rsp = a_bch.view(-1, self.T)
        inverted_bch = self.idct_2_parallel(
            self.dct_2_parallel(a_bch_rsp, K1=self.K),
            self.T
        )
        inverted_bch = inverted_bch.view(*a_bch_shape)
        if self.is_with_orig:
            inverted_bch = torch.cat((inverted_bch, a_bch), dim=1)
        print('inverted bch: ', inverted_bch.shape)
        print('生成时的误差: ', torch.sum(torch.abs(a_bch - inverted_bch)))
        return inverted_bch

    def random_transform_data(self, a_bch):
        assert self.is_with_orig
        if 'rand_0_1' in self.feature_type:
            rand_tsfm_func = torch.rand_like
        elif 'randn' in self.feature_type:
            rand_tsfm_func = torch.randn
        else:
            raise NotImplementedError('Not supported random feature type. ')
        rtn = rand_tsfm_func(a_bch).to(a_bch.device)
        if 'dot' in self.feature_type:
            rtn *= a_bch
        for i in range(self.K - 1):  # rtn has already contains a random sequence
            a_new_rand = rand_tsfm_func(a_bch).to(a_bch.device)
            if 'dot' in self.feature_type:
                a_new_rand *= a_bch
            rtn = torch.cat((rtn, a_new_rand), dim=1)
        rtn = torch.cat((a_bch, rtn), dim=1)
        return rtn

    def prepare_period_fns(self):
        assert self.inc_func is not None
        assert self.periodic_fns is not None

        # Get frequency values
        self.spat_freq_bands = []
        self.temp_freq_bands = []
        for k in range(1, self.K + 1):
            if self.inc_func == 'linear':
                a_freq = k
            elif self.inc_func == 'exp':
                a_freq = 2 ** (k - 1)
            elif self.inc_func == 'pow':
                a_freq = k ** 2
            else:
                raise NotImplementedError('Unsupported inc_func.')
            self.spat_freq_bands.append(a_freq)
            self.temp_freq_bands.append(math.pi / self.T * a_freq)

        self.spat_freq_bands = torch.tensor(self.spat_freq_bands)
        self.temp_freq_bands = torch.tensor(self.temp_freq_bands)
        print('Spatial frequency components: ', self.spat_freq_bands)
        print('Temporal frequency components: ', self.temp_freq_bands)

        # Get embed functions
        self.spat_embed_fns = []
        self.temp_embed_fns = []
        if self.is_with_orig:
            self.spat_embed_fns.append(lambda x: x)
            if self.feature_type != 'trig_spat_temp_enc':
                self.temp_embed_fns.append(lambda x, frm_idx: x)

        for freq_s, freq_t in zip(self.spat_freq_bands, self.temp_freq_bands):
            for p_fn in self.periodic_fns:
                self.spat_embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq_s: p_fn(x * freq)
                )  # STE
                self.temp_embed_fns.append(
                    lambda x, frm_idx, p_fn=p_fn, freq=freq_t: (x * p_fn(freq * (frm_idx + 1/2)))
                )  # TTE

    def trig_spat_enc_data(self, a_bch, dim):
        return torch.cat([fn(a_bch) for fn in self.spat_embed_fns], dim)

    def trig_temp_enc_data(self, a_bch, dim):
        t_len_all = a_bch.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = a_bch[:, :, t_idx, :, :].unsqueeze(2)

            new_time_list = []
            for fn in self.temp_embed_fns:
                a_new_one = fn(a_series, t_idx)
                new_time_list.append(a_new_one)
            new_time_list = torch.cat(new_time_list, dim)

            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        return rtn

    def trig_spat_temp_enc_data(self, a_bch, dim):
        spat_enc = self.trig_spat_enc_data(a_bch, dim)
        temp_enc = self.trig_temp_enc_data(a_bch, dim)
        return torch.cat((spat_enc, temp_enc), dim=dim)

    def bone_data(self, a_bch):
        bone_list = []
        bone_adj = self.get_bone_adj()
        for a_key in ntu_bone_angle_pairs:
            a_bone_value = bone_adj[a_key]
            the_joint = a_key - 1
            a_adj = a_bone_value - 1
            bone_diff = (a_bch[:, :3, :, the_joint, :] -
                         a_bch[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
            bone_list.append(bone_diff)
        bone_list = torch.cat(bone_list, dim=3)
        return bone_list

    def get_bone_adj(self):
        if 'ntu' in self.data_type:
            return ntu_bone_adj
        else:
            raise NotImplementedError('Unsupported bone adj. ')

    def get_bone_angle_pairs(self):
        if 'ntu' in self.data_type:
            return ntu_bone_angle_pairs
        else:
            raise NotImplementedError('Unsupported bone angle pairs. ')

    def process_data(self, a_dataset, save_name):
        N = len(a_dataset)
        a_dataloader = self.get_dataloader(a_dataset, self.bch_sz)
        if 'dct_tsfm' in self.feature_type:
            fp_sp_shape = (N, 6, self.T, self.V, self.M) if self.is_with_orig \
                else (N, 3, self.T, self.V, self.M)
        elif 'rand_0_1' in self.feature_type or 'randn' in self.feature_type:
            fp_sp_shape = (N, 3*(self.K+1), self.T, self.V, self.M)
        elif self.feature_type == 'trig_temp_enc' or self.feature_type == 'trig_spat_enc':
            if self.is_with_orig:
                fp_sp_shape = (N, 3 * (self.K + 1), self.T, self.V, self.M)
            else:
                fp_sp_shape = (N, 3 * self.K, self.T, self.V, self.M)
        elif self.feature_type == 'trig_spat_temp_enc':
            if self.is_with_orig:
                fp_sp_shape = (N, 6 * self.K + 3, self.T, self.V, self.M)
            else:
                fp_sp_shape = (N, 6 * self.K, self.T, self.V, self.M)
        else:
            raise NotImplementedError('Unsupported feature type. ')

        fp_sp = open_memmap(
            save_name,
            dtype='float32',
            mode='w+',
            shape=fp_sp_shape)  # concat原始data.

        for bch_idx, a_bch in enumerate(a_dataloader):
            a_bch = a_bch[0].to('cuda')
            if 'dct_tsfm' in self.feature_type:
                processed_bch = self.discrete_cosine_transform_data(a_bch)
            elif 'rand_0_1' in self.feature_type or 'randn' in self.feature_type:
                processed_bch = self.random_transform_data(a_bch)
            elif self.feature_type == 'trig_spat_enc':
                processed_bch = self.trig_spat_enc_data(a_bch, dim=1)
            elif self.feature_type == 'trig_temp_enc':
                processed_bch = self.trig_temp_enc_data(a_bch, dim=1)
            elif self.feature_type == 'trig_spat_temp_enc':
                processed_bch = self.trig_spat_temp_enc_data(a_bch, dim=1)
            else:
                raise NotImplementedError('Unsupported feature type. ')

            fp_sp[bch_idx * self.bch_sz:(bch_idx + 1) * self.bch_sz] = processed_bch.cpu().numpy()
            print('a bch shape: ', bch_idx, a_bch.shape, 'processed bch: ', processed_bch.shape)

            if self.data_status == 'train':
                the_data = self.trn_data
            elif self.data_status == 'val':
                the_data = self.val_data
            another_bch_piece = the_data[bch_idx * self.bch_sz:(bch_idx + 1) * self.bch_sz]
            print('another bch piece: ', another_bch_piece.shape)
            print('current bch diff: ', np.sum(np.abs(
                a_bch.cpu().numpy() -
                another_bch_piece
            )))

    def save_data(self):
        if self.feature_source == 'joint':
            feature_source_code = 'jnt'
        elif self.feature_source == 'bone':
            feature_source_code = 'bon'
        else:
            raise NotImplementedError('Unsupported feature source. ')

        with_orig_str = 'w_orig' if self.is_with_orig else 'enc_only'
        if self.data_status == 'train':
            trn_save_name = '{}/{}/{}_{}_{}_{}_{}.npy'.format(
                self.root_path, self.data_type, 'train',
                self.feature_type, feature_source_code,
                self.K, with_orig_str
            )
            print('trn_save_name: ', trn_save_name)
        elif self.data_status == 'val':
            val_save_name = '{}/{}/{}_{}_{}_{}_{}.npy'.format(
                self.root_path, self.data_type, 'val',
                self.feature_type, feature_source_code,
                self.K, with_orig_str
            )
            print('val_save_name: ', val_save_name)
            print('data status: ', self.data_status)
        else:
            raise NotImplementedError('Unsupported data status. ')

        if self.data_status == 'train':
            self.process_data(self.trn_data, trn_save_name)
        elif self.data_status == 'val':
            self.process_data(self.val_data, val_save_name)


if __name__ == '__main__':
    # data_enc_machine_1 = DataEncodingMachine('trig_spat_temp_enc', 'joint',
    #                                          data_status='train')
    # data_enc_machine_1.save_data()
    # data_enc_machine_2 = DataEncodingMachine('trig_spat_temp_enc', 'joint',
    #                                          data_status='val')
    # data_enc_machine_2.save_data()

    data_enc_machine_3 = DataEncodingMachine('rand_0_1_dot', 'joint',
                                             data_status='train')
    data_enc_machine_3.save_data()
    data_enc_machine_4 = DataEncodingMachine('rand_0_1_dot', 'joint',
                                             data_status='val')
    data_enc_machine_4.save_data()

