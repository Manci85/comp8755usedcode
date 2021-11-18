import numpy as np
import pickle
import sys


def get_energy(s):  # ctv
    index = s.sum(-1).sum(0) != 0  # select valid frames
    s = s[:, index, :]
    if len(s) != 0:
        s = s[0, :, :].std() + s[1, :, :].std() + s[2, :, :].std()  # three channels
    else:
        s = 0
    return s

energy_list = []

data_path = '../data/ntu120/xsub/train_data_joint.npy'
label_path = '../data/ntu120/xsub/train_label.pkl'

np_data = np.load(data_path, mmap_mode='r')
with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')

for idx, a_label in enumerate(label):
    if a_label == 50:
        # print('sample name: ', sample_name[idx])
        a_data = np_data[idx]
        person_1 = a_data[:, :, :, 0]  # C,T,V
        person_2 = a_data[:, :, :, 1]  # C,T,V
        # Select valid frames
        energy_1 = get_energy(person_1)
        energy_2 = get_energy(person_2)

        p1_min_max_diff = np.abs(np.max(person_1) - np.min(person_1))
        p2_min_max_diff = np.abs(np.max(person_2) - np.min(person_2))
        if energy_1 > energy_2:
            energy_list.append((1, 0))
        else:
            energy_list.append((0, 1))
# print('hello')
# np.set_printoptions(threshold=sys.maxsize)
# print('energy list: \n', np.array(energy_list).reshape(-1, 2))
energy_labels = np.array(energy_list).reshape(-1, 2)
for idx, a_energy_label in enumerate(energy_list):
    print(idx+1, a_energy_label)
