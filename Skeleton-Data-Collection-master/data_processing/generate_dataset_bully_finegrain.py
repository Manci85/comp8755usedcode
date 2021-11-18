import os
import glob

from numpy.lib.format import open_memmap

from action_info import action_bully_list
from utils.io_utils import get_all_subdirs
import numpy as np
from tqdm import tqdm
import math
import pickle as pkl

from visualize_saved_npy import post_visualize
import datetime


def get_left_or_right_dates():
    left_pos_dates = []
    right_pos_dates = []

    for a_year in range(2021, 2022):
        for a_month in range(5, 6+1):
            for a_day in range(31+1):
                try:
                    a_date = datetime.datetime(a_year, a_month, a_day).strftime('%Y-%m-%d')
                except ValueError:
                    continue
                if a_month == 5 or (a_month == 6 and a_day <= 14):
                    left_pos_dates.append(a_date)
                else:
                    right_pos_dates.append(a_date)

    return left_pos_dates, right_pos_dates


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



class GenInputData:
    def __init__(self):
        self.exp_dirs = \
            list(get_immediate_subdirectories(
                '/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Collected-Data/Collected-Data'
            ))
        self.left_pos_dates, self.right_pos_dates = get_left_or_right_dates()

        self.train_cameras = ['Azure-1', 'Azure-2', 'Azure-3']
        # self.train_cameras = ['Azure-1']
        self.val_cameras = ['Azure-4', 'Azure-5']
        self.gen_action_label_dict()
        self.save_data('train')
        self.save_data('val')

    def gen_action_label_dict(self):
        count_ = 0
        self.action_label_dict = {}
        for a_i in range(1, 2+1):
            for g_i in range(1, 40+1):
                action_code = 'G{}A{}'.format(g_i, a_i)
                # if action_code in asymmetric_action_codes:

                # 判断这个动作是不是bullying动作
                if action_code in action_bully_list:
                    self.action_label_dict[action_code] = count_
                    count_ += 1

    @ staticmethod
    def pre_normalization(data, zaxis=[0, 1], xaxis=[12, 5],
                          to_pad_null=True):
        data = data / 3000
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

        if to_pad_null:
            # print('pad the null frames with the previous frames')
            for i_s, skeleton in enumerate(s):  # pad
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

        # print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            main_body_center = skeleton[0][:, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

        # print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
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

        # print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
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
        return data

    def save_data(self, data_type='train'):
        if data_type == 'train':
            tgt_cameras = self.train_cameras
            input_save_name = os.path.join('/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/data_processing/test_feeding_data',
                                           'trn_data_bly_fgr.npy')
            label_save_name = os.path.join('/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/data_processing/test_feeding_data',
                                           'trn_label_bly_fgr.pkl')
        elif data_type == 'val':
            tgt_cameras = self.val_cameras
            input_save_name = os.path.join('/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/data_processing/test_feeding_data',
                                           'val_data_bly_fgr.npy')
            label_save_name = os.path.join('/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/data_processing/test_feeding_data',
                                           'val_label_bly_fgr.pkl')
        else:
            raise RuntimeError('Unsupported. ')

        label_list = []
        npy_file_list = []
        for a_exp_dir in self.exp_dirs:
            for a_trn_dir in tgt_cameras:
                a_tgt_cmr_path = os.path.join(a_exp_dir, a_trn_dir)
                if os.path.exists(a_tgt_cmr_path):
                    tgt_dir = get_all_subdirs(os.path.join(a_exp_dir, a_trn_dir), 'Skeleton')
                    if tgt_dir is None:
                        continue
                    new_npy_file_list = glob.glob(os.path.join(tgt_dir, '*.npy'))
                    if len(new_npy_file_list) == 0:
                        new_npy_file_list = glob.glob(os.path.join(tgt_dir, 'skeleton_npy', '*.npy'))
                    npy_file_list += new_npy_file_list
                else:
                    continue

        # import random
        # random.shuffle(npy_file_list)  # debug
        # all_np_tensor = np.zeros((len(npy_file_list), 3, 300, 32, 2))

        print("npy file list", len(npy_file_list))
        tgt_file_name_num = 0
        # Check how many rows
        for an_idx, a_npy_f_p in enumerate(npy_file_list):
            file_name = os.path.split(a_npy_f_p)[-1].split('-')[0]
            if file_name in action_bully_list:
                tgt_file_name_num += 1

        all_np_tensor = open_memmap(
            input_save_name,
            dtype='float32',
            mode='w+',
            shape=(tgt_file_name_num, 3, 300, 32, 2))

        print('label num: ', len(self.action_label_dict.keys()))
        is_smaller_count = 0
        is_bigger_count = 0

        idx_count = 0

        saved_sklt_num = 0
        for an_idx, a_npy_f_p in enumerate(npy_file_list):
            if an_idx % 100 == 0:
                print('Current idx: ', an_idx, 'total len: ', len(npy_file_list))
            date_dir = a_npy_f_p.split(os.sep)[7][:10]

            is_left = True
            if date_dir in self.right_pos_dates:
                is_left = False

            a_npy_f = np.load(a_npy_f_p)
            file_name = os.path.split(a_npy_f_p)[-1].split('-')[0]

            # single label
            # an_label = self.action_label_dict[file_name]

            _, C, T, V, M = a_npy_f.shape
            if T < 5:
                continue

            two_skeletons = np.array((-a_npy_f[0, 0, :, :, :],
                                      -a_npy_f[0, 2, :, :, :],
                                      -a_npy_f[0, 1, :, :, :]))

            # two_skeletons = np.array((a_npy_f[0, 0, :, :, :],
            #                           a_npy_f[0, 1, :, :, :],
            #                           a_npy_f[0, 2, :, :, :]))

            skeleton_0 = two_skeletons[:C, :T, :V, 0]
            skeleton_1 = two_skeletons[:C, :T, :V, 1]

            if is_left:
                try:
                    is_0_positive = np.all(skeleton_0[0, 10, :V] < skeleton_1[0, 10, :V])
                except IndexError:
                    is_0_positive = np.all(skeleton_0[0, 2, :V] < skeleton_1[0, 2, :V])
            else:
                try:
                    is_0_positive = np.all(skeleton_0[0, 10, :V] > skeleton_1[0, 10, :V])
                except IndexError:
                    is_0_positive = np.all(skeleton_0[0, 2, :V] > skeleton_1[0, 2, :V])

            # 保证 skeleton 0 是主动位
            # if not is_0_positive:
            #     skeleton_0, skeleton_1 = skeleton_1, skeleton_0

            if is_0_positive:
                is_smaller_count += 1
            else:
                is_bigger_count += 1

            # 更改一下skeleton的shape, 使其能够被加入
            skeleton_0 = np.expand_dims(np.expand_dims(skeleton_0, axis=0), axis=-1)
            skeleton_1 = np.expand_dims(np.expand_dims(skeleton_1, axis=0), axis=-1)

            # 如果这个action pair是一个bullying动作
            if file_name in action_bully_list:
                # 创立一个空的action pair, 里面要包含两个人的数据
                skeleton_zero = np.zeros((1, 3, 300, 32, 2))
                skeleton_zero[:, :, :skeleton_0.shape[2], :, 0:1] = skeleton_0
                skeleton_zero[:, :, :skeleton_0.shape[2], :, 1:2] = skeleton_1
                skeleton_zero = self.pre_normalization(skeleton_zero)
                all_np_tensor[idx_count] = skeleton_zero

                # 标签插入一次, 因为只有一个action
                a_label = self.action_label_dict[file_name]
                if is_0_positive:
                    a_label = [a_label, 1, 0]
                else:
                    a_label = [a_label, 0, 1]
                label_list.append(a_label)

                # 加了1个动作的skeleton
                idx_count += 1
                saved_sklt_num += 1

                # Check的时候保存到新文件夹
                if saved_sklt_num != 0 and saved_sklt_num % 5000 == 0:
                    post_visualize(all_np_tensor[idx_count-1:idx_count, :C, :T, :V, :2],
                                   save_name=
                                   f'skeleton_check/bullying/{file_name}_{an_idx}_{a_label}.mp4')
                    # post_visualize(all_np_tensor[idx_count+1:idx_count+2, :C, :T, :V, :1],
                    #                save_name=f'skeleton_check/{file_name}_{an_idx}.mp4')
                    print('file name', file_name, 'an idx: ', an_idx)

        for a_data in all_np_tensor:
            print('sum of tmp data: ', np.sum(a_data))

        print('sum: ', np.sum(all_np_tensor))
        # np.save(input_save_name, all_np_tensor)
        print('label_list: ', label_list)
        with open(label_save_name, 'wb') as a_pkl:
            pkl.dump(('sth', label_list), a_pkl, protocol=pkl.HIGHEST_PROTOCOL)

        print('is smaller count: ', is_smaller_count, 'is bigger count: ', is_bigger_count)


if __name__ == '__main__':
    gen_input_data = GenInputData()
