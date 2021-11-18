import torch.nn as nn
import torch
import torch.nn.functional as functional

if __name__ == '__main__':
    tmp = torch.arange(10).float().reshape(1, 2, 5).repeat(2, 1, 1)
    tmp[:, 1] = tmp[:, 1] + 10
    print('tmp 1: ', tmp)
    tmp = functional.interpolate(tmp, size=8, mode='linear')
    print('tmp: ', tmp)
    print('tmp: ', tmp.shape)
