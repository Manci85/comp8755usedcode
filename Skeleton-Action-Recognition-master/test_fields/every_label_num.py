from collections import defaultdict

import numpy as np
import pickle

datapath = ''
labelpath = ''
dataset = 'ntu120/xsub'

tgt_action = 51

data_path = '../data/{}/val_data_joint.npy'.format(dataset)
label_path = '../data/{}/val_label.pkl'.format(dataset)

data = np.load(data_path, mmap_mode='r')
with open(label_path, 'rb') as f:
    labels = pickle.load(f, encoding='latin1')

total_num = defaultdict(int)
for a_label in labels[1]:
    total_num[a_label+1] += 1

print(total_num)
