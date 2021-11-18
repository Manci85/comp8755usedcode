import numpy as np
import torch

a_1 = np.array([1, 2, 3])
a_2 = np.array([4, 5, 6])

cross = np.cross(a_1, a_2)
print('cross: ', cross)

b_1 = torch.tensor([1, 2, 3])
b_2 = torch.tensor([4, 5, 6])

cross_torch = torch.cross(b_1, b_2)
print('cross torch: ', cross_torch)


