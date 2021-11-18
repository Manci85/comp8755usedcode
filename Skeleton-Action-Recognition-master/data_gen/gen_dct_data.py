import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import TensorDataset
import math


benchmarks = {
    # 'ntu': ('ntu/xview', 'ntu/xsub'),
    # 'ntu': ('ntu/xview',),
    'ntu120': ('ntu120/xset',),
    # 'ntu120': ('ntu120/xsub',),
    'kinetics': ('kinetics',),
    'anubis102': ('data/anu-all/',),
}

parts = ('train', 'val')
# parts = {'train'}

multires = 3
to_include_input = True
inc_func = 'linear'


class Embedder_DCT:
    def __init__(self):
        self.frm_len = 300.0
        self.kwargs = {
            'include_input': to_include_input,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            # 'periodic_fns': [torch.sin, torch.cos],
            'periodic_fns': [torch.cos],
        }

        self.create_embedding_fn()

    def get_out_dim(self):
        return int(2. ** self.kwargs['num_freqs']) + 1

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x, y: x)  # with x
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = []
        for k in range(1, N_freqs+1):
            if inc_func == 'linear':
                a_freq = k
            elif inc_func == 'exp':
                a_freq = 2 ** (k-1)
            elif inc_func == 'pow':
                a_freq = k ** 2
            else:
                raise NotImplementedError('Unsupported inc_func.')

            freq_bands.append(math.pi / self.frm_len * a_freq)  # This is DCT

        freq_bands = torch.tensor(freq_bands)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # this is DCT
                embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * p_fn(freq * (frm_idx + 1/2))))
                # 不用cosine进行伸缩
                # embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * (freq * (frm_idx + 1/2))))
                # 不使用x
                # embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (torch.ones_like(x) * p_fn(freq * (frm_idx + 1/2))))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        t_len_all = inputs.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = inputs[:, :, t_idx, :, :].unsqueeze(2)
            # new_time_list = torch.cat([fn(a_series, t_idx) for fn in self.embed_fns], dim)  # DCT

            # To try positional encoding
            new_time_list = []
            for fn in self.embed_fns:
                a_new_one = fn(a_series, t_idx)
                new_time_list.append(a_new_one)
            new_time_list = torch.cat(new_time_list, dim)

            # To sum encodes
            # new_time_list = None
            # for fn in self.embed_fns:
            #     if new_time_list is None:
            #         new_time_list = fn(a_series, t_idx)
            #     else:
            #         new_time_list += fn(a_series, t_idx)

            # print('new_time_list: ', new_time_list.squeeze())
            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        return rtn


def gen_dct_data(fea_type, dataset_type):
    if fea_type == 'joint':
        # save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
        #             'data/{}/{}_data_jnt_dct_{}.npy'
        save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                    'MS-G3D/data/{}/{}_jnt_dct_{}_{}_{}.npy'
        # save_name = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_jnt_dct_{}_{}_{}.npy'
    elif fea_type == 'bone':
        save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                    'MS-G3D/data/{}/{}_bon_dct_{}_{}_{}.npy'
        # save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
        #             'MS-G3D/data/{}/{}_bon_dct_{}_{}_{}.npy'
    else:
        raise NotImplementedError
    print('save name: ', save_name)

    for benchmark in benchmarks[dataset_type]:
        for part in parts:
            if fea_type == 'joint':
                data_path = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                               'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '../data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data' \
                #                 '/{}/{}_data_joint.npy'.format(benchmark, part)
                # data = np.load('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
                #                '/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
            elif fea_type == 'bone':
                data_path = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                               'MS-G3D/data/{}/{}_data_bone.npy'.format(benchmark, part)
                # data_path = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                #             'MS-G3D/data/{}/{}_data_bone.npy'.format(benchmark, part)
            else:
                raise NotImplementedError

            data = np.load(data_path.format(benchmark, part), mmap_mode='r')
            print('load data path: ', data_path)

            # For debugging
            # data = data[:2]
            # print('data: ', data[0, 0, :, 0, 0])
            # data = torch.ones([1, 1, 4, 1, 1])

            N, C, T, V, M = data.shape
            print('data shape: ', data.shape)

            inc_input_str = 'w_orig' if to_include_input else 'enc_only'
            print('saving to: ', save_name.format(benchmark, part, inc_func, multires, inc_input_str))

            # Output shape
            fp_sp_shape = (N, (multires+1)*3, T, V, M) if to_include_input else (N, multires*3, T, V, M)

            fp_sp = open_memmap(
                save_name.format(benchmark, part, inc_func, multires, inc_input_str),
                dtype='float32',
                mode='w+',
                shape=fp_sp_shape)
                # shape=(N, multires*3, T, V, M))  # Not include the original input
                # shape=(N, multires*6 + 3, T, V, M))
                # shape=(N, 3, T, V, M))

            print(benchmark, part)

            # an_embed = Embedder()
            an_embed = Embedder_DCT()
            load_bch_sz = 3000

            a_dataset = TensorDataset(torch.tensor(data))
            a_dataloader = torch.utils.data.DataLoader(
                dataset=a_dataset,
                batch_size=load_bch_sz,
                shuffle=False,
                num_workers=4,
                drop_last=False
            )

            for bch_idx, a_bch in enumerate(a_dataloader):
                print('bch idx: ', bch_idx, 'a bch: ', a_bch[0].shape)
                a_piece = an_embed.embed(a_bch[0].to('cuda'), dim=1).cpu().numpy()
                print('piece shape: ', a_piece.shape)
                # print('a_piece: ', a_piece[0, 0, :, 0, 0])
                # print('a_piece: ', a_piece[0, 1, :, 0, 0])
                # print('a_piece: ', a_piece[0, 2, :, 0, 0])
                fp_sp[bch_idx * load_bch_sz:(bch_idx + 1) * load_bch_sz] = a_piece
            print('fp_sp: ', fp_sp.shape)


if __name__ == '__main__':
    gen_dct_data('joint', 'ntu120')
