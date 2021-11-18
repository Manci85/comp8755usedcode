import numpy as np
import torch

from data_gen.data_encoding_machine import DataEncodingMachine

if __name__ == '__main__':
    orig_data_path = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data//ntu/xsub/train_data_joint.npy'
    recon_data_path = '/media/zhenyue-qin/New Volume/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data//ntu/xsub/trn_ntu_freq_back_jnt_300.npy'

    orig_data = DataEncodingMachine.load_data(orig_data_path)[:500]
    recon_data = DataEncodingMachine.load_data(recon_data_path)
    print('recon sum: ', np.sum(orig_data))
    # print('orig 1: ', orig_data[1])
    # print('recon 1: ', recon_data[1])

    print('误差: ', np.sum(np.abs(orig_data - recon_data)))
