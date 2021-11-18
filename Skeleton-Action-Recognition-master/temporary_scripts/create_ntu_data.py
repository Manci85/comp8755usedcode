import numpy as np
from numpy.lib.format import open_memmap

data_train_p = '/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/train_data_joint_bone_angle_cangle_arms_legs.npy'
data_val_p = '/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_data_joint_bone_angle_cangle_arms_legs.npy'

data_train_joint = np.load(data_train_p, mmap_mode='r')[:, :3, :, :, :]
data_val_joint = np.load(data_val_p, mmap_mode='r')[:, :3, :, :, :]
print('data train joint: ', data_train_joint.shape)
print('data val joint: ', data_val_joint.shape)

data_train_bone = np.load(data_train_p, mmap_mode='r')[:, 3:6, :, :, :]
data_val_bone = np.load(data_val_p, mmap_mode='r')[:, 3:6, :, :, :]
print('data train bone: ', data_train_bone.shape)
print('data val bone: ', data_val_bone.shape)

print('generating train_joint_save_name')
train_joint_save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/train_data_joint.npy'
new_train_data_joint = open_memmap(
    train_joint_save_name,
    dtype='float32',
    mode='w+',
    shape=(len(data_train_joint), 3, 300, 25, 2)
)
new_train_data_joint[:, :, :, :, :] = data_train_joint[:, :, :, :, :]

print('generating train_bone_save_name')
train_bone_save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/train_data_bone.npy'
new_train_data_bone = open_memmap(
    train_bone_save_name,
    dtype='float32',
    mode='w+',
    shape=(len(data_train_bone), 3, 300, 25, 2)
)
new_train_data_bone[:, :, :, :, :] = data_train_bone[:, :, :, :, :]

print('generating val_joint_save_name')
val_joint_save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_data_joint.npy'
new_val_data_joint = open_memmap(
    val_joint_save_name,
    dtype='float32',
    mode='w+',
    shape=(len(data_val_joint), 3, 300, 25, 2)
)
new_val_data_joint[:, :, :, :, :] = data_val_joint[:, :, :, :, :]

print('generating val_bone_save_name')
val_bone_save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_data_bone.npy'
new_val_data_bone = open_memmap(
    val_bone_save_name,
    dtype='float32',
    mode='w+',
    shape=(len(data_val_bone), 3, 300, 25, 2)
)
new_val_data_bone[:, :, :, :, :] = data_val_bone[:, :, :, :, :]

