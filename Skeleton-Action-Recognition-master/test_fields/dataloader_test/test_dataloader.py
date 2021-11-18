import os

import h5py
import numpy as np

from data_gen.preprocess_zq import pre_normalization

max_body_true = 2
max_body_kinect = 4
num_joint = 25

# 这些人的ID是用来训练的
training_subjects = set([
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
])

# 这些setup的ID是用来训练的
# Even numbered setups (2,4,...,32) used for training
training_setups = set(range(2, 33, 2))


def read_skeleton_filter(path):
    with open(path, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(path, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(path)
    # Create single skeleton tensor: (M, T, V, C)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # To (C,T,V,M)
    data = data.transpose(3, 1, 2, 0)
    return data


def get_unbatched_skeleton_data(file_list, ignored_sample_path, benchmark, part, joint_type):
    is_bone = False
    if joint_type == 'bone':
        is_bone = True
    h5_save = h5py.File('ntu120_{}_{}_{}.hdf5'.format(benchmark, part, joint_type), 'w')
    ignored_samples = []
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    sample_label = []
    sample_paths = []
    filename_list = []
    for folder, filename in sorted(file_list):
        if filename in ignored_samples:
            continue

        path = os.path.join(folder, filename)
        filename_list.append(filename.replace('.skeleton', ''))
        setup_loc = filename.find('S')
        subject_loc = filename.find('P')
        action_loc = filename.find('A')
        setup_id = int(filename[(setup_loc + 1):(setup_loc + 4)])
        subject_id = int(filename[(subject_loc + 1):(subject_loc + 4)])
        action_class = int(filename[(action_loc + 1):(action_loc + 4)])

        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xset':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError(f'Unsupported benchmark: {benchmark}')

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError(f'Unsupported dataset part: {part}')

        if issample:
            sample_paths.append(path)
            sample_label.append(action_class - 1)   # to 0-indexed

        # if len(sample_label) > 100:
        #     break

    for i, s in enumerate(sample_paths):
        if i % 100 == 0:
            print('i: ', i)
        data = read_xyz(s, max_body=max_body_kinect, num_joint=num_joint)
        data = np.expand_dims(data, axis=0)
        data = pre_normalization(data, to_pad_null=False, to_use_bone=is_bone)
        data = data[0]
        filename_save = filename_list[i]
        h5_save['{}/data'.format(filename_save)] = data
        h5_save['{}/label'.format(filename_save)] = sample_label[i]
    h5_save.close()


if __name__ == '__main__':
    print('Current path: ', os.getcwd())
    ntu120_ignore_list = '../../../../Datasets/nturgbd_raw/NTU_RGBD120_samples_with_missing_skeletons.txt'
    ntu120_part1_path = '../../../../Datasets/nturgbd_raw/nturgb+d_skeletons/'
    ntu120_path2_path = '../../../../Datasets/nturgbd_raw/nturgb+d_skeletons120/'

    benchmark = 'xsub'
    part = 'val'
    joint_type = 'bone'

    # Combine skeleton file paths
    file_list = []
    for folder in [ntu120_part1_path, ntu120_path2_path]:
        for path in os.listdir(folder):
            file_list.append((folder, path))

    get_unbatched_skeleton_data(file_list, ntu120_ignore_list, benchmark=benchmark, part=part,
                                joint_type=joint_type)
