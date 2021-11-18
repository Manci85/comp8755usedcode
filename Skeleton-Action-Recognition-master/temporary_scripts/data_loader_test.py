import torch
import torch.utils.data
from torch.utils.data import TensorDataset
import numpy as np

if __name__ == '__main__':
    # tmp_data = torch.arange(1, 105).unsqueeze(1).repeat(1, 10)
    # tmp_data = TensorDataset(tmp_data)
    # a_dataloader = torch.utils.data.DataLoader(
    #     dataset=tmp_data,
    #     batch_size=10,
    #     shuffle=False,
    #     num_workers=4,
    #     drop_last=False
    # )
    # for bch_idx, a_bch in enumerate(a_dataloader):
    #     a_bch = a_bch[0]
    #     print('bch idx: ', bch_idx)
    #     print('a bch: ', a_bch.shape)
    #     print('a bch: ', a_bch)

    bch_sz = 10
    data_p = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/' \
             '2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_data_joint.npy'
    data = np.load(data_p, mmap_mode='r')[:12]

    a_dataset = TensorDataset(torch.tensor(np.array(data)))
    a_dataloader = torch.utils.data.DataLoader(
        dataset=a_dataset,
        batch_size=bch_sz,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    for bch_idx, a_bch in enumerate(a_dataloader):
        a_bch = a_bch[0]
        load_bch_sz = a_bch.shape[0]
        print('bch idx: ', bch_idx)
        diff = torch.sum(torch.abs(
            a_bch -
            torch.tensor(data[bch_idx * bch_sz:(bch_idx + 1) * bch_sz]).to(a_bch.device))
        )
        print('difference: ', diff)




