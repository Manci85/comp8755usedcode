import sys
sys.path.extend(['../'])

from tqdm import tqdm

from data_gen.rotation import *

ntu_bone_dict = {
    1: 2, 2: 21, 3: 21, 4: 3, 5: 21, 6: 5,
    7: 6, 8: 7, 9: 21, 10: 9, 11: 10, 12: 11,
    13: 1, 14: 13, 15: 14, 16: 15, 17: 1, 18: 17,
    19: 18, 20: 19, 22: 23, 21: 21, 23: 8, 24: 25, 25: 12
}

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4],
                      to_pad_null=True, to_use_bone=False):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    if to_pad_null:
        print('pad the null frames with the previous frames')
        for i_s, skeleton in enumerate(tqdm(s)):  # pad
            if skeleton.sum() == 0:
                print(i_s, ' has no skeleton')
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                if person[0].sum() == 0:
                    index = (person.sum(-1).sum(-1) != 0)
                    tmp = person[index].copy()
                    person *= 0
                    person[:len(tmp)] = tmp
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        if person[i_f:].sum() == 0:
                            rest = len(person) - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                            s[i_s, i_p, i_f:] = pad
                            break

    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    if to_use_bone:
        for a_joint_0, a_joint_1 in ntu_skeleton_bone_pairs:
            v0 = a_joint_0 - 1
            v1 = a_joint_1 - 1
            data[:, :, :, v0, :] = data[:, :, :, v0, :] - data[:, :, :, v1, :]
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
