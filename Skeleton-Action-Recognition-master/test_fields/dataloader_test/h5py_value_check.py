import torch
import numpy as np
import h5py
import pickle

tmp_1 = h5py.File('ntu120_xsub_val_bone.hdf5', 'r')
tmp_2 = np.load('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_data_bone.npy',
                mmap_mode='r')
label_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'
with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')

tgt_id = 5
key_0 = sorted(tmp_1.keys())[tgt_id]
print('key 0: ', key_0)
print(tmp_1[key_0]['data'][()][0][0][0])
print(tmp_1[key_0]['label'][()])
print('######')
print(tmp_2[tgt_id][0][0][0])
print('@@@@@')
print(sample_name[0])
print('hello world')
