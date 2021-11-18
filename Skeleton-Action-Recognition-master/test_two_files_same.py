import numpy as np
import torch
from torch.utils.data import TensorDataset


class DataIdentityChecker:
    def __init__(self):
        self.data_p1 = ''
        self.data_p2 = ''

    @staticmethod
    def get_dataloader(a_data, bch_sz=1500):
        a_dataset = TensorDataset(torch.tensor(a_data))
        a_dataloader = torch.utils.data.DataLoader(
            dataset=a_dataset,
            batch_size=bch_sz,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        return a_dataloader

# data_p1 = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_data_joint.npy'
# data_p2 = '/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_data_joint_bone_angle_cangle_arms_legs.npy'

data_p1 = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_trig_spat_temp_enc_jnt_3_w_orig.npy'
data_p2 = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_jnt_dct_linear_8_w_orig.npy'

data_num = 50000
data_piece_num = 50000

print('data p1: ', data_p1)
print('data p2: ', data_p2)

data_1 = np.load(data_p1, mmap_mode='r')[:data_num][:, 12:21, :, :, :]
data_2 = np.load(data_p2, mmap_mode='r')[:data_num][:, 3:12, :, :, :]

diff_idx_list = []
for a_f_idx in range(len(data_1)):
    a_data_1 = data_1[a_f_idx]
    a_data_2 = data_2[a_f_idx]

    data_diff = np.sum(np.abs(a_data_1 - a_data_2))
    if data_diff > 0:
        print('ATTENTION!!! a f idx is different: ', a_f_idx)
        diff_idx_list.append(a_f_idx)
    else:
        if a_f_idx % 100 == 0:
            print('a f idx is the same: ', a_f_idx)

print('diff_idx_list: ', diff_idx_list)
