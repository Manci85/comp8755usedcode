import numpy as np

tmp_file = '/media/zhenyue-qin/Backup Plus1/Glow_Denoising/train_glow.sh'

with open(tmp_file, 'rb') as a_file:
    print('a file: ', a_file.read().decode('ISO-8859-1').encode("utf-8").decode('utf-8'))