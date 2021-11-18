import os
import torch

src_path = "/media/zhenyue-qin/Samsung_T5/Data/CelebA"
tar_path = "/media/zhenyue-qin/Seagate Expansion Drive/Yang_Liu/Data/CelebA/Img/img_align_celeba"
img_folder = 'Img/img_align_celeba'
data = torch.load(os.path.join(src_path, 'processed.pt'))
# paths_GT = data['train']
paths_GT = data['val']
for i in range(len(paths_GT)):
    # print(paths_GT[i][0])
    file_path = os.path.join(src_path, img_folder, paths_GT[i][0])
    # new_path = os.path.join(tar_path, paths_GT[i][0])
    # if not os.path.isfile(new_path):
        # print(file_path)
    command = 'mv "%s" "%s"'%(file_path, tar_path)
    os.system(command)