import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data


def subvideo_collate(a_batch):
    frame_per_block = 15
    overlap_frame = 5
    for a_instance in a_batch:
        video = a_instance['data'][()]
        video_len = video.shape[1]


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.hdf5_data = h5py.File(file_path, 'r')
        self.filenames = sorted(self.hdf5_data.keys())

    def __getitem__(self, index):
        # get data
        a_filename = self.filenames[index]
        return self.hdf5_data[a_filename]['data'], self.hdf5_data[a_filename]['label']

    def __len__(self):
        return len(self.hdf5_data)


if __name__ == '__main__':
    a_hdf5_dataset = HDF5Dataset('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/test_fields/dataloader_test/ntu120_xsub_train_bone.hdf5')
    data_loader = data.DataLoader(a_hdf5_dataset)
    for a_data in data_loader:
        print('a data: ', a_data)
        break