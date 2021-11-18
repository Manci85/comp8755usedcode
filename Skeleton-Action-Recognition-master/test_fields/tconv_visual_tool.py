import torch
import matplotlib.pyplot as plt
import os
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}

matplotlib.rc('font', **font)

tconv_path_0 = '../eval/tensor_analysis/sklt_0-layer_0-tconv_signal.pt'
tconv_path_1 = '../eval/tensor_analysis/sklt_0-layer_1-tconv_signal.pt'
tconv_path_2 = '../eval/tensor_analysis/sklt_0-layer_2-tconv_signal.pt'
tconv_path_3 = '../eval/tensor_analysis/sklt_0-layer_3-tconv_signal.pt'
tconv_path_4 = '../eval/tensor_analysis/sklt_0-layer_4-tconv_signal.pt'
tconv_path_5 = '../eval/tensor_analysis/sklt_0-layer_5-tconv_signal.pt'
tconv_path_6 = '../eval/tensor_analysis/sklt_0-layer_6-tconv_signal.pt'
tconv_path_7 = '../eval/tensor_analysis/sklt_0-layer_7-tconv_signal.pt'
tconv_path_8 = '../eval/tensor_analysis/sklt_0-layer_8-tconv_signal.pt'
tconv_path_9 = '../eval/tensor_analysis/sklt_0-layer_9-tconv_signal.pt'

a_tensor_0 = torch.load(tconv_path_0)[0]
# print('a tensor 0: ', torch.sum(a_tensor_0, dim=0).transpose(1, 0))

a_tensor_1 = torch.load(tconv_path_1)[0]
# print('a tensor 1: ', torch.sum(a_tensor_1, dim=0).transpose(1, 0))

a_tensor_2 = torch.load(tconv_path_2)[0]
# print('a tensor 2: ', torch.sum(a_tensor_2, dim=0).transpose(1, 0))

a_tensor_3 = torch.load(tconv_path_3)[0]
a_tensor_4 = torch.load(tconv_path_4)[0]
a_tensor_5 = torch.load(tconv_path_5)[0]
a_tensor_6 = torch.load(tconv_path_6)[0]
a_tensor_7 = torch.load(tconv_path_7)[0]
a_tensor_8 = torch.load(tconv_path_8)[0]
a_tensor_9 = torch.load(tconv_path_9)[0]

a_tensor_0 = torch.sum(a_tensor_0, dim=0).transpose(1, 0) * 10
a_tensor_1 = torch.sum(a_tensor_1, dim=0).transpose(1, 0)
a_tensor_2 = torch.sum(a_tensor_2, dim=0).transpose(1, 0)
a_tensor_3 = torch.sum(a_tensor_3, dim=0).transpose(1, 0)
a_tensor_4 = torch.sum(a_tensor_4, dim=0).transpose(1, 0)
a_tensor_5 = torch.sum(a_tensor_5, dim=0).transpose(1, 0)
a_tensor_6 = torch.sum(a_tensor_6, dim=0).transpose(1, 0)
a_tensor_7 = torch.sum(a_tensor_7, dim=0).transpose(1, 0)
a_tensor_8 = torch.sum(a_tensor_8, dim=0).transpose(1, 0)
a_tensor_9 = torch.sum(a_tensor_9, dim=0).transpose(1, 0)
# a_tensor_2 = torch.sum(a_tensor_2, dim=0).transpose(1, 0)

# for a_tensor_curve in a_tensor:
#     plt.plot(a_tensor_curve.cpu().numpy())

jnt_idx = 11
plt.plot(a_tensor_0[jnt_idx].cpu().numpy(), label="signal")
plt.plot(a_tensor_1[jnt_idx].cpu().numpy(), label="layer-1")
plt.plot(a_tensor_2[jnt_idx].cpu().numpy(), label="layer-2")
plt.plot(a_tensor_3[jnt_idx].cpu().numpy(), label="layer-3")
# plt.plot(a_tensor_4[jnt_idx].cpu().numpy(), label="layer-4")
# plt.plot(a_tensor_5[jnt_idx].cpu().numpy(), label="layer-5")
# plt.plot(a_tensor_6[jnt_idx].cpu().numpy(), label="layer-6")
# plt.plot(a_tensor_7[jnt_idx].cpu().numpy(), label="layer-7")
# plt.plot(a_tensor_8[jnt_idx].cpu().numpy(), label="layer-8")
# plt.plot(a_tensor_9[jnt_idx].cpu().numpy(), label="layer-9")
# plt.plot(a_tensor_2[jnt_idx].cpu().numpy(), label="dct-enc")

x = list(range(300))
xi = list(range(300))
# plt.xticks(xi, x)

save_name = tconv_path_0.split(os.sep)[-1] + '_' + tconv_path_1.split(os.sep)[-1] + '.png'
# show legend
plt.legend()

for xc in list(range(0, 300, 5)):
    plt.axvline(x=xc, linestyle='dashed', linewidth=1)

plt.savefig(os.path.join('../eval/tensor_analysis/', save_name), dpi=600)
# plt.savefig('tmp.png', dpi=300)
# plt.show()
