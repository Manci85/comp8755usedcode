import torch

cuda_0 = torch.cuda.get_device_name(0)
print('cuda 0: ', cuda_0)

cuda_1 = torch.cuda.get_device_name(1)
print('cuda 1: ', cuda_1)

cuda_2 = torch.cuda.get_device_name(2)
print('cuda 2: ', cuda_2)

cuda_3 = torch.cuda.get_device_name(3)
print('cuda 3: ', cuda_3)

