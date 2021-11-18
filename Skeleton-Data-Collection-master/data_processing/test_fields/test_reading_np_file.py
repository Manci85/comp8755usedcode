import numpy as np

np_file = '/media/zhenyue-qin/Elements/Data-Collection/Skeleton-Data-Collection/data_processing/' \
          'test_feeding_data/trn_ind_data_non_violent.npy'

tmp_data = np.load(np_file)

for a_data in tmp_data:
    print('sum of tmp data: ', np.sum(a_data))
