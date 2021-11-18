import numpy as np
import pickle
import math
import scipy.fftpack
import torch
import torch_dct as dct
import matplotlib.pyplot as plt

from utils_dir.utils_math import dct_2_no_sum_parallel, gen_dct_on_the_fly
from utils_dir.utils_visual import azure_kinect_post_visualize


class DCT_InverseVisual:
    def __init__(self):
        self.data_path = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_jnt_dct_linear_8_w_orig.npy'
        self.label_path = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl'
        self.use_mmap = True

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load label
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

    @staticmethod
    def add_noise_to_skeleton(a_skeleton):
        noise = torch.randn_like(a_skeleton)
        return a_skeleton + 0.01 * noise

    def test(self):
        from scipy.fftpack import ifft, idct
        a = ifft(np.array([30., -8., 6., -2., 6., -8.])).real
        print('a: ', a)
        b = idct(np.array([30., -8., 6., -2.]), 1) / 6
        print('b: ', b)

    def test_dct_on_the_fly(self):
        self.dct_data_path = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_jnt_dct_linear_8_w_orig.npy'
        if self.use_mmap:
            self.dct_data = np.load(self.dct_data_path, mmap_mode='r')
        else:
            self.dct_data = np.load(self.data_path)

        data = torch.tensor(self.data[785])[:3, :, :, :].unsqueeze(0)
        data_dct = torch.tensor(self.dct_data[785])[:, :, :, :].unsqueeze(0)

        self.dct_K = 8
        data = gen_dct_on_the_fly(data, K=self.dct_K)
        # data = torch.cat((data, dct_part), dim=1)
        print('data: ', data.shape)

        a_tmp_data = data[0, :, :, :, :]
        a_tmp_data_dct = data_dct[0, :, :, :, :]
        print('a tmp dct data: ', a_tmp_data.shape)
        print('a_tmp_data_dct original: ', a_tmp_data_dct.shape)
        print('two sum: ', torch.sum(torch.abs(a_tmp_data[4] - a_tmp_data_dct[4])))

    def tmp(self):
        tgt_joint_idx = 11
        sklt_id = 101
        visual_k = 0
        a_skeleton = torch.tensor(self.data[sklt_id])[visual_k*3:visual_k*3+3, :, :, 0].unsqueeze(-1)
        # a_skeleton = torch.tensor(self.data[1267])[:, :, :, 0].unsqueeze(-1)
        a_trajectory = list([x.item() for x in a_skeleton[0, :, tgt_joint_idx, 0]])
        plt.plot(a_trajectory)
        plt.show()
        print('jnt trajectory: ', a_trajectory)
        assert 0

        # noisy_skeleton = self.add_noise_to_skeleton(a_skeleton)
        azure_kinect_post_visualize(a_skeleton.unsqueeze(0), sklt_type='kinect_v2',
                                    save_name='dct-visual/{}_klt_cosine_k_{}.mp4'.format(sklt_id, visual_k))
        assert 0

        # a_skeleton = torch.tensor(self.data[4206])[:3, :, 0, 0]
        a_skeleton_rsp = a_skeleton.view(-1, a_skeleton.shape[1])
        list_len = len(a_skeleton_rsp[0])
        freq_k = 300
        # a_skeleton = torch.tensor(list(range(list_len))).unsqueeze(0).float()

        # print('original: ', a_skeleton_rsp[0])
        bch_dct_seq = self.dct_2_parallel(a_skeleton_rsp, K1=freq_k)
        # print('bch dct seq: ', bch_dct_seq[0])
        # dct_seq = self.dct_2_process(a_skeleton_rsp[0], K1=freq_k)
        # print('dct seq: ', dct_seq)
        bch_idct_seq = self.idct_2_parallel(bch_dct_seq, K=list_len)
        # print('bch idct seq: ', bch_idct_seq[0])
        # idct_seq = self.inverse_dct_2_process(dct_seq, K=list_len)
        # print('idct seq: ', idct_seq)
        recon_sklt = bch_idct_seq.view(*a_skeleton.shape)
        print('误差: ', torch.sum(torch.abs(recon_sklt - a_skeleton)))
        assert 0

        plt.plot(a_skeleton[0, :, tgt_joint_idx, 0].cpu().numpy(), label="orig")
        plt.plot(recon_sklt[0, :, tgt_joint_idx, 0].cpu().numpy(), label="dct")
        plt.legend()
        for xc in list(range(0, 300, 5)):
            plt.axvline(x=xc, linestyle='dashed', linewidth=1)

        plt.savefig('信号对比.png', dpi=200)


        # seq_x = a_skeleton[0]
        # seq_y = a_skeleton[1]
        # seq_z = a_skeleton[2]
        # a_sklt_sequence = a_skeleton[0, 0, 21, :]
        # dct_sklt_sequence = self.dct_2_process(a_sklt_sequence)
        # idct_sklt_sequence = self.inverse_dct_2_process(dct_sklt_sequence)
        # print(torch.sum(torch.abs(a_sklt_sequence - idct_sklt_sequence)))
        azure_kinect_post_visualize(recon_sklt.unsqueeze(0), sklt_type='kinect_v2',
                                    save_name='sklt_orig.mp4')

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
        basis_list = torch.tensor(basis_list)
        dot_prod = torch.einsum('ab,bc->ac', bch_seq_rsp, basis_list.transpose(1, 0))
        return dot_prod.view(-1, K1-K0)

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
        basis_list = torch.tensor(basis_list)
        dot_prod = torch.einsum('ab,bc->ac', bch_seq_rsp, basis_list.transpose(1, 0))
        bch_x0 = bch_seq_rsp[:, 0].unsqueeze(1).repeat(1, K) * 0.5
        unnorm_rst = dot_prod + bch_x0
        norm_rst = unnorm_rst * (2.0 / K)
        return norm_rst.view(-1, K)

    @staticmethod
    def dct_2_process(a_seq, K0=0, K1=None):
        N = len(a_seq)
        rtn = []
        if K1 is None:
            K1 = len(a_seq)
        for k in range(K0, K1):
            a_coe_list = []
            for i, a_x in enumerate(a_seq):
                a_new_x = a_x * math.cos(math.pi / N * (i + 0.5) * k)
                a_coe_list.append(a_new_x)
            rtn.append(np.sum(a_coe_list))
        return rtn

    def dct_2_process_tensor(self, a_tensor, K_0=0, K_1=None):
        C, T, V, M = a_tensor.shape
        a_tensor_rsp = a_tensor.view(C * V * M, T)
        for a_idx in range(C * V * M):
            print('a idx: ', a_idx)
            a_tensor_rsp[a_idx] = torch.tensor(self.dct_2_process(a_tensor_rsp[a_idx], K_0, K_1))
        return a_tensor_rsp.view(C, T, V, M)

    @staticmethod
    def inverse_dct_2_process(a_seq, K=None):
        # N是分量个数, K是序列的长度
        N = len(a_seq)
        if K is None:
            K = len(a_seq)
        x_0 = a_seq[0]
        rtn = []
        for k in range(K):
            a_sum = 0
            for n in range(1, N):
                x_n = a_seq[n]
                a_sum += x_n * math.cos((math.pi / N) * n * (k + 0.5))
            rtn.append(0.5 * x_0 + a_sum)
        return np.array(rtn) * (2 / K)

    def test_dct_inverse(self):
        a_sequence = [1, 2, 3, 4, 5, 6]
        tmp_a = self.dct_2_process(a_sequence)
        tmp_b = scipy.fftpack.dct(a_sequence, type=2)
        print('tmp a: ', tmp_a)
        print('tmp b: ', tmp_b)
        tmp_c = self.inverse_dct_2_process(tmp_a)
        print('tmp c: ', tmp_c)
        tmp_d = scipy.fftpack.idct(tmp_b, type=2)
        print('tmp d: ', tmp_d)

        x = torch.randn(200)
        X = dct.dct(x)  # DCT-II done through the last dimension
        y = dct.idct(X)  # scaled DCT-III done through the last dimension
        print('sum: ', torch.sum(torch.abs(x - y)))


if __name__ == '__main__':
    dct_inverse_visual = DCT_InverseVisual()
    # dct_inverse_visual.test_dct_on_the_fly()
    dct_inverse_visual.tmp()
