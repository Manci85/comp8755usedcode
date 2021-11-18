import torch


x = torch.arange(1 * 5 * 2).view(1, 5, 2)
print(x)
print('####')
x_repeat = torch.repeat_interleave(x, 5, dim=1)
print(x_repeat)
