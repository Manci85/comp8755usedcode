import torch
import torch_dct as dct
import scipy.fftpack
torch.manual_seed(0)

x = torch.randn(100).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1). \
    repeat(2, 3, 1, 25, 2)
print(x)
X = dct.dct(x[0, 0, :, 0, 0])
print('X: ', X)
# y = dct.idct(X)
# assert (torch.abs(x - y)).sum() < 1e-3

tmp = scipy.fftpack.dct(x.numpy(), axis=2)
# print('tmp: ', tmp)
print('tmp: ', tmp[0, 0, :, 0, 0])
