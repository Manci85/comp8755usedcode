import torch

tmp = torch.eye(5).unsqueeze(1).unsqueeze(-1).unsqueeze(0).repeat(1, 1, 3, 1, 2)

print('tmp: \n', tmp[:, :, :, :, 1][0, :, 0, :])
