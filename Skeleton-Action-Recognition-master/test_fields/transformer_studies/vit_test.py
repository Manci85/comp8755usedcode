import torch
import timm
from vit_pytorch import ViT
import torch.nn as nn

img = torch.randn(1, 3, 224, 224)

# v = ViT(
#     image_size=256,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )
# _ = v(img)
#
# model_names = timm.list_models(pretrained=True)
# print('model names: ', model_names)
#
# m = timm.create_model('vit_base_patch16_224', pretrained=True)
# m.eval()
# print(m)
# preds = m(img)
# print('hello')

from torchvision import datasets
from torchvision.transforms import transforms

traindir = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/test_fields/imgnet_test_imgs'
# traindir = '/data1/imagenet/ILSVRC2012/train/'
# valdir = '/media/zhenyue-qin/Elements/Datasets/ImageNet-2012/val'
# valdir = '/media/zhenyue-qin/Elements/Datasets/ImageNet-2012/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
batch_size = 50

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=12, pin_memory=True)

# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=batch_size, shuffle=False,
#     num_workers=6, pin_memory=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

rand_input = torch.randn((50, 196, 768)).to('cuda')
NUM_FINETUNE_CLASSES = 120
model = timm.create_model('vit_base_patch16_224', pretrained=True).to('cuda')
model.patch_embed = Identity()
print(model)
for a_data, a_label in train_loader:
    a_data = a_data.to('cuda')
    sb = model(rand_input)
    print('sb: ', torch.argmax(sb, dim=-1))
    assert 0






