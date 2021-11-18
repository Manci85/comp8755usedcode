import torch
import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import TensorDataset
import math


benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    # 'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'ntu120': ('ntu120/xset',),
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}


class Embedder_DCT:
    def __init__(self):
        multires = 5
        self.frm_len = 300.0
        self.kwargs = {
            'include_input': True,
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
            embed_fns.append(lambda x, y: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = []
        for k in range(1, N_freqs+1):
            freq_bands.append(math.pi / self.frm_len * k)
        freq_bands = torch.tensor(freq_bands)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * p_fn(freq * (frm_idx + 1/2))))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        inputs_confident = inputs[:, 2, :, :, :].unsqueeze(1)
        inputs = inputs[:, :2, :, :, :]
        t_len_all = inputs.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = inputs[:, :, t_idx, :, :].unsqueeze(2)
            new_time_list = torch.cat([fn(a_series, t_idx) for fn in self.embed_fns], dim)
            # print('new_time_list: ', new_time_list.squeeze())
            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        rtn = torch.cat((rtn, inputs_confident), dim=1)
        return rtn


class Embedder:
    def __init__(self):
        multires = 2
        self.kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.create_embedding_fn()

    def get_out_dim(self):
        return int(2. ** self.kwargs['num_freqs']) + 1

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: x * p_fn(freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim)


def gen_nerf_data(fea_type, dataset_type):
    if fea_type == 'joint':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_jnt_nerf.npy'
        # save_name = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_data_jnt_nerf.npy'
    elif fea_type == 'bone':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_bon_nerf.npy'
        # save_name = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_data_bon_nerf.npy'
    else:
        raise NotImplementedError
    print('save name: ', save_name)
    multires = 2
    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    for benchmark in benchmarks[dataset_type]:
        for part in parts:
            if fea_type == 'joint':
                data_path = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                               'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data' \
                #                 '/{}/{}_data_joint.npy'.format(benchmark, part)
                # data = np.load('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
                #                '/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
            elif fea_type == 'bone':
                data_path = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                            'Datasets/{}/{}_data_bon_nerf_orig.npy'.format(benchmark, part)
            else:
                raise NotImplementedError

            data = np.load(data_path.format(benchmark, part), mmap_mode='r')
            print('load data path: ', data_path)

            # For debugging
            # data = data[:100]
            # data = torch.ones([1, 1, 4, 1, 1])

            N, C, T, V, M = data.shape
            print('data shape: ', data.shape)

            fp_sp = open_memmap(
                save_name.format(benchmark, part),
                dtype='float32',
                mode='w+',
                shape=(N, 13, T, V, M))

            print(benchmark, part)

            an_embed = Embedder_DCT()
            load_bch_sz = 2000

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
                if fea_type == 'bone':
                    a_bch[0] = a_bch[0][:, 3:6, :, :, :]
                a_piece = an_embed.embed(a_bch[0].to('cuda'), dim=1).cpu().numpy()
                # print('piece shape: ', a_piece.shape)
                # print('a_piece: ', a_piece[0, 0, :, 0, 0])
                # print('a_piece: ', a_piece[0, 1, :, 0, 0])
                # print('a_piece: ', a_piece[0, 2, :, 0, 0])
                fp_sp[bch_idx * load_bch_sz:(bch_idx + 1) * load_bch_sz] = a_piece
            print('fp_sp: ', fp_sp.shape)


if __name__ == '__main__':
    gen_nerf_data('bone', 'kinetics')
