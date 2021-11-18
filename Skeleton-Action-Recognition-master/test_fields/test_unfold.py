import torch

B, C, H, W = 2, 3, 5, 5
x = torch.arange(B*C*H*W).view(B, C, H, W).float()

print(x)

kernel_h, kernel_w = 2, 2
stride = 1

a_unfold = torch.nn.Unfold(kernel_size=(2, 2),
                           dilation=(1, 1),
                           stride=(1, 1),
                           padding=(0, 0))

# patches = x.unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
patches = a_unfold(x)
print(patches.shape)
