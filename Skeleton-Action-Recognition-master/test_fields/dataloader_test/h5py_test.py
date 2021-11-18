import h5py
import numpy as np

f = h5py.File('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/test_fields/dataloader_test/skeleton_data.hdf5', 'r')

print(f['S001C001P001R001A001']['data'])
