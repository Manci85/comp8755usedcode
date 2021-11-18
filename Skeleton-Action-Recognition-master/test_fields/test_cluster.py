import numpy as np
from sklearn.cluster import KMeans
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

from test_fields.test_relabel_mutual_actions import energy_labels

np_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-02-18-ntu120_xsub_jnt_1psn_get_model_features/' \
          '2021-02-20T15-03-43/model_features.npy'

a_npy = np.load(np_path, mmap_mode='r')

# PCA
pca = PCA(n_components=2)
a_npy = pca.fit_transform(a_npy)

# kmeans
kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20).fit(a_npy)
labels = kmeans.labels_

if len(labels) % 2 == 1:
    labels = labels[:-1]
np.set_printoptions(threshold=sys.maxsize)
# print('labels: ', labels)
print(labels.reshape(-1, 2))
labels_reshape = labels.reshape(-1, 2)

correct = 0
for idx, pred in enumerate(labels_reshape):
    if pred[1] == energy_labels[idx][1]:
        correct += 1
    if pred[0] == energy_labels[idx][0]:
        correct += 1
print('correct: ', correct, 'total: ', 2 * len(labels_reshape))

# print('energy labels: \n', energy_labels)
