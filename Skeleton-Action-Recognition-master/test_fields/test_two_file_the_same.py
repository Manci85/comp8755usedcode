import numpy as np

data_p_1 = '/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/anu-all/trn_data_jnt_dct_3_enc_only.npy'
data_p_2 = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/anubis/frt-bck/train_trig_temp_enc_jnt_3_enc_only.npy'

data_a = np.load(data_p_1, mmap_mode='r')[:100]
data_b = np.load(data_p_2, mmap_mode='r')[:100]

tmp = np.sum(np.abs(data_a - data_b))
print('difference: ', np.mean(np.abs(data_a - data_b)))

np_1_p = '/media/zhenyue-qin/local/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/anubis/frt-bck/train_data_joint.npy'
np_2_p = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_data_joint.npy'
np_3_p = '/media/zhenyue-qin/local/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/anubis/frt-bck/val_data_joint.npy'
np_4_p = '/mnt/80F484E6F484E030/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_data_joint.npy'

# data_1 = np.load(np_1_p, mmap_mode='r')[:200][:, :, :, [0, 3, 7, 14, 21, 25], 0]
# data_2 = np.load(np_2_p, mmap_mode='r')[:200][:, :, :, [0, 3, 7, 11, 15, 19], 0]
# data_3 = np.load(np_3_p, mmap_mode='r')[:200][:, :, :, [0, 3, 7, 14, 21, 25], 0]
# data_4 = np.load(np_4_p, mmap_mode='r')[:200][:, :, :, [0, 3, 7, 11, 15, 19], 0]

data_1 = np.load(np_1_p, mmap_mode='r')[:40000][:, :, :, :, 0]
data_2 = np.load(np_2_p, mmap_mode='r')[:40000][:, :, :, :, 0]
data_3 = np.load(np_3_p, mmap_mode='r')[:40000][:, :, :, :, 0]
data_4 = np.load(np_4_p, mmap_mode='r')[:40000][:, :, :, :, 0]

# print('avg diff 1: ', np.mean(data_2 / data_1))
# print('avg diff 2: ', np.mean(data_4 / data_3))

mean_1 = np.mean(data_1)
mean_2 = np.mean(data_2)
mean_3 = np.mean(data_3)
mean_4 = np.mean(data_4)

max_1 = np.max(data_1)
max_2 = np.max(data_2)
max_3 = np.max(data_3)
max_4 = np.max(data_4)

min_1 = np.min(data_1)
min_2 = np.min(data_2)
min_3 = np.min(data_3)
min_4 = np.min(data_4)

print('mean data 1: ', mean_1)
print('mean data 2: ', mean_2)
print('mean data 3: ', mean_3)
print('mean data 4: ', mean_4)

print('max data 1: ', max_1, 'min data 1: ', min_1)
print('max data 2: ', max_2, 'min data 2: ', min_2)
print('max data 3: ', max_3, 'min data 3: ', min_3)
print('max data 4: ', max_4, 'min data 4: ', min_4)

print('mean 1 division: ', mean_2 / mean_1)
print('mean 2 division: ', mean_4 / mean_3)

print('min 1 division: ', min_2 / min_1)
print('max 1 division: ', max_2 / max_1)
print('min 2 division: ', min_4 / min_3)
print('max 2 division: ', max_4 / max_3)
# print('difference: ', np.sum(np.abs(data_1 - data_2)))
