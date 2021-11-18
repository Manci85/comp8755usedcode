import torch

def tmp():
    batch_size = 5
    F = 7
    N = 5

    # Create m matrix and mask some values
    m = torch.randn(N, N)
    m[0, 4] = 0
    m[2, 1] = 0
    m[4, 3] = 0
    print(m)

    # Get zero indices
    mask_idx = (m == 0).nonzero()

    # Create x and mask same indices
    x = torch.randn(batch_size, N, N, F)
    x[:, mask_idx[:, 0], mask_idx[:, 1], :] = 0

    # Print some results
    print(x[0, :, :, 0])
    print(x[1, :, :, 0])
    print(x[2, :, :, 3])


def tmp2():
    tmp_value = torch.randn((3, 3, 3))
    # print('tmp value: \n', tmp_value.fill_diagonal_(5))
    tmp_value = tmp_value.unsqueeze(-1).repeat(1, 1, 1, 3)
    sth = torch.zeros_like(tmp_value) + 1
    tmp_value[0, 0, :, :] *= torch.eye(3)
    print('tmp value: \n', tmp_value[0, 0, :, :])


if __name__ == '__main__':
    tmp2()
