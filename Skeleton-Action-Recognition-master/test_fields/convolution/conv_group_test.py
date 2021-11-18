import torch

if __name__ == '__main__':
    data_in = torch.arange(1, 13, dtype=torch.float32).view(4, -1).unsqueeze(0).repeat(2, 1, 1)
    data_in[0] = -data_in[0]
    data_in = data_in.unsqueeze(0)
    print(data_in)

    tmp_weight = torch.nn.Parameter(
        torch.arange(12, dtype=torch.float32).view(4, 1, 3, 1)
    )
    conv_group = torch.nn.Conv2d(in_channels=2, out_channels=4,
                                 kernel_size=(3, 1), groups=2, bias=False)
    conv_group.weight = tmp_weight
    # conv_group.weight = torch.nn.Parameter(torch.zeros_like(tmp_weight))
    print('tmp weight: ', tmp_weight, tmp_weight.shape)
    out = conv_group(data_in)
    print(out)
