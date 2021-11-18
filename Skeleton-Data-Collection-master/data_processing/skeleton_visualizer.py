from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import os


# bone_pairs = (
#     (0, 1), (1, 2), (2, 3), (3, 26),
#     (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),
#     (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
#     (2, 1), (1, 0), (0, 22), (0, 18), (22, 23), (23, 24), (24, 25),
#     (18, 19), (19, 20), (20, 21)
# )
from hyperparameters import start_off, end_off
from kinect_info import bone_pairs


def generate_random_string(a_len):
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(a_len))


def visualize(frames, save_name=None):
    bones = bone_pairs
    def animate(skeletons):
        # Skeleton shape is 3*25. 3 corresponds to the 3D coordinates. 25 is the number of joints.
        ax.clear()

        ax.set_xlim([-3000, 1000])
        ax.set_ylim([-3000, 1000])
        ax.set_zlim([-3000, 1000])

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

        # person 1
        k = 0
        color_list = ('blue', 'orange', 'cyan', 'purple')
        # color_list = ('blue', 'blue', 'cyan', 'purple')
        color_idx = 0

        while k < skeletons.shape[0]:
            for i, j in bones:
                joint_locs = skeletons[:, [i, j]]
                # plot them
                ax.plot(-joint_locs[k + 0], -joint_locs[k + 2], -joint_locs[k + 1], color=color_list[color_idx])

            k += 3
            color_idx = (color_idx + 1) % len(color_list)

        # Rotate
        # X, Y, Z = axes3d.get_test_data(0.1)
        # ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
        #
        # # rotate the axes and update
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)

        if save_name is None:
            title = 'Action Visualization'
        else:
            title = os.path.split(save_name)[-1]
        plt.title(title)
        skeleton_index[0] += 1
        return ax

    for an_entry in range(1):

        if isinstance(an_entry, tuple) and len(an_entry) == 2:
            index = int(an_entry[0])
            pred_idx = int(an_entry[1])
        else:
            index = an_entry
        # get data
        skeletons = np.copy(frames[index])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])

        # print(f'Sample index: {index}\nAction: {action_class}-{action_name}\n')  # (C,T,V,M)

        # Pick the first body to visualize
        skeleton1 = skeletons[..., 0]  # out (C,T,V)
        # make it shorter
        shorter_frame_start = 0
        shorter_frame_end = 120
        skeleton1 = skeleton1[:, shorter_frame_start:shorter_frame_end, :]
        if skeletons.shape[-1] > 1:
            skeleton2 = np.copy(skeletons[..., 1])  # out (C,T,V)
            # make it shorter
            skeleton2 = skeleton2[:, shorter_frame_start:shorter_frame_end, :]
            # print('max of skeleton 2: ', np.max(skeleton2))
            skeleton_frames_2 = skeleton2.transpose(1, 0, 2)
        else:
            skeleton_frames_2 = None

        skeleton_index = [0]
        skeleton_frames_1 = skeleton1.transpose(1, 0, 2)

        if skeleton_frames_2 is None:
            ani = FuncAnimation(fig, animate,
                                skeleton_frames_1,
                                interval=150)
        else:
            ani = FuncAnimation(fig, animate,
                            np.concatenate((skeleton_frames_1, skeleton_frames_2), axis=1),
                            interval=150)

        if save_name is None:
            save_name = 'tmp_skeleton_video.mp4'
        print('skeleton save name: ', save_name)
        ani.save(save_name, dpi=100, writer='ffmpeg')
        plt.close('all')

def get_wrong_sample_idx(f_path):
    idxes = []
    preds = []
    with open(f_path) as file_in:
        lines = []
        for line in file_in:
            line_brk = line.split(',')
            lines.append(line)
            idxes.append(int(line_brk[0]))
            preds.append(int(line_brk[1]))
    return idxes, preds


def get_all_start_end_time(the_csv):
    action_time_list = []
    the_csv = pd.read_csv(the_csv, sep=',')
    for an_idx, a_row in the_csv.iterrows():
        action_id = a_row[0]
        active_ = a_row[1]
        orientation_ = a_row[2]
        start_time = a_row[-3]
        end_time = a_row[-2]
        action_time_list.append((action_id, start_time, end_time, orientation_, active_))
    return action_time_list


def convert_time_to_second(hms, apm=None):
    if ' ' not in hms:
        hms = hms.split(':')
    else:
        if apm is None:
            hms = hms.split(' ')[1].split(':')
        else:
            hms = hms.split(':')
    hms = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
    if apm is not None:
        if apm == 'PM':
            hms += 12 * 3600
    return hms


def catch_relative_component(a_row, target):
    for an_element in a_row:
        if target in an_element:
            return an_element


def get_action_clips(log_csv, skeleton_csv, a_save_root_path):
    start_offset = start_off
    end_offset = end_off

    a_save_root_path = a_save_root_path.replace(':', 'h')
    if not os.path.exists(a_save_root_path):
        os.makedirs(a_save_root_path)
    # file name count dict
    file_name_count_dict = defaultdict(int)

    at_list = get_all_start_end_time(log_csv)
    skeleton_csv = pd.read_csv(skeleton_csv, sep=' ', header=None)
    prev_time = None

    at_idx = 0  # start line
    action_skeleton_dict = {}

    cur_at = at_list[at_idx]

    print('cur at: ', cur_at)

    # Update actions
    cur_action = cur_at[0]
    cur_start_time = convert_time_to_second(cur_at[1]) + start_offset
    cur_end_time = convert_time_to_second(cur_at[2]) + end_offset
    forward_backward = cur_at[3]
    active_passive = cur_at[4]
    action_skeleton_dict[cur_action] = {}

    is_acting = False

    print('skeleton_csv: ', len(skeleton_csv))
    count_ = 0
    for index, row in skeleton_csv.iterrows():
        # if count_ % 20000 == 0:
        #     print('cur at: ', cur_at)
        # count_ += 1
        try:
            body_id = catch_relative_component(row, 'Body')
        except Exception as e:
            print('Exception: ', e)
            print('Exception at: ', cur_at)
            print('row id: ', index, 'total len: ', len(skeleton_csv))
            print('Problem row: ', row)
            continue
        # the_time = convert_time_to_second(row[1], row[2])
        the_time = convert_time_to_second(row[1])
        if prev_time is None:
            prev_time = the_time
            continue
        # print('prev time: ', prev_time, '; current start time: ', cur_start_time,
        #       'current end time: ', cur_end_time)
        if (prev_time <= cur_start_time and the_time >= cur_start_time) or \
                ((prev_time > cur_start_time and the_time > cur_start_time) and not is_acting):
        # if True and not is_acting:
            # print('new body id: ', body_id)
            a_new_frame = []
            frame_idx = 0
            is_acting = True

        if is_acting:
            if frame_idx % 32 == 0 and a_new_frame != []:
                min_body_id = body_id
                min_diff = float('inf')
                if len(action_skeleton_dict[cur_action].keys()) > 1:
                    for a_body_id in action_skeleton_dict[cur_action].keys():
                        # print('something here: ', action_skeleton_dict)
                        last_frames = action_skeleton_dict[cur_action][a_body_id][-1]
                        a_diff = np.mean(np.abs(np.array(a_new_frame) - np.array(last_frames)))
                        # print('a diff: ', a_diff)
                        if a_diff < min_diff:
                            min_body_id = a_body_id
                            min_diff = a_diff
                else:
                    if body_id not in action_skeleton_dict[cur_action]:
                        action_skeleton_dict[cur_action][body_id] = []

                action_skeleton_dict[cur_action][min_body_id].append(a_new_frame)
                a_new_frame = []

            x_cor = float(catch_relative_component(row, 'X').replace('X', '').replace(':', ''))
            y_cor_z_cor = catch_relative_component(row, 'Y').replace('Y', '').replace(':', '').split('Z')
            y_cor = float(y_cor_z_cor[0])
            z_cor = float(y_cor_z_cor[1])
            x_y_z = (x_cor, y_cor, z_cor)
            a_new_frame.append(x_y_z)
            frame_idx += 1

        if (prev_time < cur_end_time and the_time >= cur_end_time) or \
            (index >= len(skeleton_csv)-1) or \
                ((prev_time > cur_end_time and the_time > cur_end_time) and is_acting):
            is_acting = False

            try:
                key_list = list(action_skeleton_dict[cur_action].keys())
                if len(key_list) > 2:
                    print('More than 2 skeleton IDs')
                    frame_len_list = np.array(list([len(action_skeleton_dict[cur_action][an_id])
                                                    for an_id in key_list]))
                    frame_len_argmax = np.argsort(frame_len_list)[::-1][:2]
                    key_list = [key_list[frame_len_argmax[0]], key_list[frame_len_argmax[1]]]
                body_id_1, body_id_2 = key_list
                # body_id_1 = list(action_skeleton_dict[cur_action].keys())[0]
                to_visualize_1 = torch.tensor(action_skeleton_dict[cur_action][body_id_1]).\
                    permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
                to_visualize_2 = torch.tensor(action_skeleton_dict[cur_action][body_id_2]). \
                    permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
                frame_len = min(to_visualize_1.shape[2], to_visualize_2.shape[2])
                to_visualize = torch.cat((to_visualize_1[:, :, -frame_len:, :, :],
                                          to_visualize_2[:, :, -frame_len:, :, :]), dim=-1)
            except:
                print('Error at: ', cur_action, forward_backward, active_passive)
                # Update actions
                at_idx += 1
                cur_at = at_list[at_idx]
                cur_action = cur_at[0]
                cur_start_time = convert_time_to_second(cur_at[1]) + start_offset
                cur_end_time = convert_time_to_second(cur_at[2]) + end_offset
                forward_backward = cur_at[3]
                active_passive = cur_at[4]
                action_skeleton_dict[cur_action] = {}
                continue

            # print('to_visualize: ', to_visualize_1.shape)
            # visualize(to_visualize)
            a_file_name = '{}_{}_{}'.format(
                cur_action.replace(': ', '-').replace(' ', '-'), forward_backward, active_passive
            )
            file_name_count = file_name_count_dict[a_file_name]
            to_save_path_ani = os.path.join(a_save_root_path, 'skeleton_animations', a_file_name + '-' + str(file_name_count)).replace(':', 'h')
            to_save_path_npy = os.path.join(a_save_root_path, 'skeleton_npy', a_file_name + '-' + str(file_name_count)).replace(':', 'h')
            if not os.path.exists(os.path.join(a_save_root_path, 'skeleton_animations')):
                os.makedirs(os.path.join(a_save_root_path, 'skeleton_animations'))
            if not os.path.exists(os.path.join(a_save_root_path, 'skeleton_npy')):
                os.makedirs(os.path.join(a_save_root_path, 'skeleton_npy'))
            visualize(to_visualize, to_save_path_ani + '.mp4')
            # print('to visualize: ', to_visualize.shape)
            with open(to_save_path_npy + '.npy', 'wb') as np_f:
                np.save(np_f, to_visualize.cpu().numpy())
            # break
            file_name_count_dict[a_file_name] += 1

            # Update actions
            at_idx += 1
            cur_at = at_list[at_idx]
            cur_action = cur_at[0]
            cur_start_time = convert_time_to_second(cur_at[1]) + start_offset
            cur_end_time = convert_time_to_second(cur_at[2]) + end_offset
            forward_backward = cur_at[3]
            active_passive = cur_at[4]
            action_skeleton_dict[cur_action] = {}

        prev_time = the_time
        # except:
        #     print('this row goes wrong: ', cur_at)
        #     continue


if __name__ == '__main__':
    # action_log_csv = 'action_logs/2021-05-05T19-10-08_action_log.csv'
    # action_time_dict = get_all_start_end_time(action_log_csv)
    # 修改这里
    log_csv = '/media/zhenyue-qin/Elements/Data-Collection/Skeleton-Data-Collection/time_logger/saved_csv/2021-05-30T14-40-38_action_log.csv'
    # 修改这里
    # skeleton_csv = '/media/zhenyue-qin/One Touch/Azure-3-Data/skeletonData-2021-05-30--14-38-46.txt'
    skeleton_csv = '/media/zhenyue-qin/One Touch1/Azure1-Data/skeletonData-2021-05-30--14-38-34.txt'

    # a_csv = pd.read_csv('raw_data/skeletonData-2021-05-05--19-02-07.txt',
    #                     sep=' ', header=None)

    a_random_str = generate_random_string(4)
    save_root_path = f'../2021-05-30T14:30-16:30-Azure-1-Skeleton_{a_random_str}'
    assert save_root_path[3:13] in log_csv and save_root_path[3:13] in skeleton_csv

    get_action_clips(log_csv, skeleton_csv, save_root_path)

