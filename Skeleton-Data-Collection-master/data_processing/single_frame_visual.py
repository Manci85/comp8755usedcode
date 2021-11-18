import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kinect_info import bone_pairs
from tqdm import tqdm


def center_normalize_skeleton(s):
    # T, C, V = s.shape
    main_body_center = s[:, :, 1:2].copy()
    s = s - main_body_center
    return s

def post_visualize(frames, save_name=None):
    bones = bone_pairs
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
            # skeleton_frames_1 = center_normalize_skeleton(skeleton_frames_1)
            tgt_skeleton = skeleton_frames_1
        else:
            # skeleton_frames_1 = center_normalize_skeleton(skeleton_frames_1)
            # skeleton_frames_2 = center_normalize_skeleton(skeleton_frames_2)

            tgt_skeleton = np.concatenate((skeleton_frames_1, skeleton_frames_2))

        a_frame = tgt_skeleton[0]
        ax.clear()

        # ax.set_xlim([-1000, 1000])
        # ax.set_ylim([-1500, 600])
        # ax.set_zlim([-3000, 3000])

        ax.set_xlim([-3000, 3000])
        ax.set_ylim([-3000, 3000])
        ax.set_zlim([-3000, 3000])

        ax.set_xticklabels(['x'])
        ax.set_yticklabels(['y'])
        ax.set_zticklabels(['z'])

        # person 1
        k = 0
        color_list = ('blue', 'orange', 'cyan', 'purple')
        # color_list = ('blue', 'blue', 'cyan', 'purple')
        color_idx = 0

        while k < a_frame.shape[0]:
            for i, j in bones:
                joint_locs = a_frame[:, [i, j]]
                # plot them
                # ax.plot(joint_locs[k+0], joint_locs[k+1], joint_locs[k+2], color=color_list[color_idx])
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
        # ax.view_init(0, 0)
        skeleton_index[0] += 1
        plt.savefig('tmp_frame.png', dpi=200)


def visualize_npy(a_npy_dir):
    save_dir = os.path.join(a_npy_dir, 'test_np')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    a_npy_dir = os.path.join(a_npy_dir, '*.npy')
    npy_file_list = glob.glob(a_npy_dir)
    for a_npy_path in npy_file_list:
        print('a npy path: ', a_npy_path)
        np_name = os.path.split(a_npy_path)[-1].replace('.npy', '') + '.mp4'
        a_npy_f = np.load(a_npy_path)
        # save_name = os.path.join(save_dir, np_name)
        save_name = None
        post_visualize(a_npy_f, save_name)
        assert 0


if __name__ == '__main__':
    npy_dir_path = '/media/zhenyue-qin/Elements/Data-Collection/Collected-Data/2021-05-17T14:00-16:00/Azure-1/2021-05-17T14h00-16h00-Azure-1-Skeleton_BK1R'
    visualize_npy(npy_dir_path)
