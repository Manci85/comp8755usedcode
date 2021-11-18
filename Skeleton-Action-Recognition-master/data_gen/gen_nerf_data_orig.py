import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}
# parts = {'val'}

multires = 3
to_include_input = True
inc_func = 'linear'

class Embedder:
    def __init__(self):
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
        if to_include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        # if self.kwargs['log_sampling']:
        #     freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        # else:
        #     freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        # print('freq bands: ', freq_bands)

        freq_bands = []
        for k in range(1, N_freqs + 1):
            if inc_func == 'linear':
                a_freq = k
            elif inc_func == 'exp':
                a_freq = 2 ** (k - 1)
            elif inc_func == 'pow':
                a_freq = k ** 2
            else:
                raise NotImplementedError('Unsupported inc_func.')
            freq_bands.append(a_freq)
        freq_bands = torch.tensor(freq_bands)
        print('freq band: ', freq_bands)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # Nerf
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: x * p_fn(freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim)


def gen_nerf_data(fea_type, dataset_type):
    if fea_type == 'joint':
        save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_ste_jnt_{}_{}_{}.npy'
        # save_name = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
        #                     'MS-G3D/data/{}/{}_ste_jnt_{}_{}_{}.npy'
        # save_name = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_data_jnt_nerf.npy'
        # save_name = '/mnt/80F484E6F484E030/' \
        #             'Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_data_jnt_nerf_k_3.npy'
    elif fea_type == 'bone':
        save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_ste_bon_{}_{}_{}.npy'
    else:
        raise NotImplementedError
    print('save name: ', save_name)

    for benchmark in benchmarks[dataset_type]:
        for part in parts:
            if fea_type == 'joint':  # /media/zhenyue-qin/New Volume
                data_path = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                            'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                #             'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '../data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data' \
                #                '/{}/{}_data_joint.npy'.format(benchmark, part)
            elif fea_type == 'bone':
                data_path = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                            'MS-G3D/data/{}/{}_data_bone.npy'.format(benchmark, part)
                # data_path = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                #             'MS-G3D/data/{}/{}_data_bone.npy'.format(benchmark, part)
                # data_path = '../data/{}/{}_data_bone.npy'.format(benchmark, part)
            else:
                raise NotImplementedError

            data = np.load(data_path.format(benchmark, part), mmap_mode='r')
            print('load data path: ', data_path)

            # For debugging
            # data = data[:100]
            # data = torch.ones([1, 1, 4, 1, 1])

            N, C, T, V, M = data.shape
            print('data shape: ', data.shape)

            inc_input_str = 'w_orig' if to_include_input else 'enc_only'
            print('saving to: ', save_name.format(benchmark, part, inc_func, multires, inc_input_str))

            # fp_sp_shape = (N, 3 + 2*3*(multires), T, V, M) if to_include_input else (N, 2*3*(multires), T, V, M)
            fp_sp_shape = (N, 3 + 3*(multires), T, V, M) if to_include_input else (N, 3*(multires), T, V, M)
            fp_sp = open_memmap(
                save_name.format(benchmark, part, inc_func, multires, inc_input_str),
                dtype='float32',
                mode='w+',
                shape=fp_sp_shape)  # concat原始data.
                # shape=(N, 2*3*(multires), T, V, M))  # 不使用原始data.

            print(benchmark, part)

            an_embed = Embedder()
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
                # print('piece shape: ', a_piece.shape)
                # print('a_piece: ', a_piece[0, 0, :, 0, 0])
                # print('a_piece: ', a_piece[0, 1, :, 0, 0])
                # print('a_piece: ', a_piece[0, 2, :, 0, 0])
                fp_sp[bch_idx * load_bch_sz:(bch_idx + 1) * load_bch_sz] = a_piece
            print('fp_sp: ', fp_sp.shape)


if __name__ == '__main__':
    gen_nerf_data('joint', 'ntu120')
