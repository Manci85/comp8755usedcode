import torch

tmp_1 = torch.ones([2, 2, 2, 3]).float()
tmp_2 = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                      [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]]).float()
print('tmp_2: ', tmp_2.shape)

tmp_3 = tmp_2.unsqueeze(1).repeat((1, 2, 1, 1))

mul_ = torch.einsum('nctu,nuv->nctv', tmp_1, tmp_2)
mul_ = torch.einsum('nctu,ncuv->nctv', tmp_1, tmp_3)
# print('mul_: \n', tmp_3[1][0])
# print('mul_: \n', mul_[1])
# print('mul_: \n', mul_.shape)


tmp_a = torch.arange(6).view((2, 3))
print('tmp a: ', tmp_a)
tmp_b = torch.arange(12).view((2, 2, 3))
print('tmp b: ', tmp_b)
tmp_c = torch.einsum('ve,bte->btve', tmp_a, tmp_b)
print('tmp c: ', tmp_c)

inp_1 = torch.arange(12).view((2, 2, 3))
inp_2 = torch.arange(6).view(3, 2)
tmp_d = torch.einsum('btv,ve->bte', inp_1, inp_2)
print('tmp d: ', tmp_d)

print('New Test 3: ')
tmp_1 = torch.tensor((1, 2, 3)).unsqueeze(-1)
tmp_2 = torch.tensor((4, 5, 6)).unsqueeze(0)

tmp_1_2 = torch.einsum('ij,jk->ik', tmp_1, tmp_2)
print('tmp 1 2: ', tmp_1_2)
