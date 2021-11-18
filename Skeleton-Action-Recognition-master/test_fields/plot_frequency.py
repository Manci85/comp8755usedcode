import numpy as np
import pickle
from scipy.fftpack import fft, dct
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def plot_frequency(a_list):
    the_dct = dct(a_list, 2)
    print('the dct: ', the_dct)
    plt.bar(list(range(len(the_dct))), the_dct,
            edgecolor='black')
    plt.show()

if __name__ == '__main__':
    the_dataset = 'ntu/xsub/'

    data_path = '../data/{}/val_data_joint.npy'.format(the_dataset)
    label_path = '../data/{}/val_label.pkl'.format(the_dataset)

    data = np.load(data_path, mmap_mode='r')
    with open(label_path, 'rb') as f:
        labels = pickle.load(f, encoding='latin1')

    for idx, a_data in enumerate(data):
        print(a_data.shape)
        plot_frequency(a_data[0, :30, 24, 0])
        assert 0
