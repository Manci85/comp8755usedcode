import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

bone_pairs = (
    (0, 1), (1, 2), (2, 3), (3, 26),
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),
    (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
    (2, 1), (1, 0), (0, 22), (0, 18), (22, 23), (23, 24), (24, 25),
    (18, 19), (19, 20), (20, 21)
)


def visualize(frames):
    bones = bone_pairs
    def animate(skeletons):
        # 这里skeleton是3*25, 3是3d坐标, 25是joint的数量.
        ax.clear()

        ax.set_xlim([-1400, -400])
        ax.set_ylim([0, 1200])
        ax.set_zlim([0, 1200])

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

        # person 1
        k = 0
        color_list = ('blue', 'orange', 'cyan', 'purple')
        color_idx = 0
        while k < skeletons.shape[0]:
            for i, j in bones:
                joint_locs = skeletons[:, [i, j]]
                # plot them
                ax.plot(joint_locs[k+0], joint_locs[k+1], joint_locs[k+2], color=color_list[color_idx])

            k += 3
            color_idx = (color_idx + 1) % len(color_list)

        title = 'Madhawa Skeleton'
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

        save_name = '{}_{}'.format('action_class', 'action_name')

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

        ani = FuncAnimation(fig, animate, skeleton_frames_1, interval=300)

        ani.save('tmp.avi', dpi=200, writer='ffmpeg')


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


if __name__ == '__main__':
    a_csv = pd.read_csv('/media/zhenyue-qin/Elements/Downloads/Miscellaneous/test.txt',
                        sep=' ', header=None)

    frames = []
    a_new_frame = []
    frame_idx = 0
    for index, row in a_csv.iterrows():
        x_cor = float(row[2].replace('X', '').replace(':', ''))
        y_cor_z_cor = row[3].replace('Y', '').replace(':', '').split('Z')
        y_cor = float(y_cor_z_cor[0])
        z_cor = float(y_cor_z_cor[1])

        body_id = row[0]
        if body_id == 'BodyID:1':
            if frame_idx % 32 == 0:
                frames.append(a_new_frame)
                a_new_frame = []

            x_y_z = (x_cor, y_cor, z_cor)
            a_new_frame.append(x_y_z)
            frame_idx += 1

    frames = torch.tensor(frames[1:]).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
    print(frames.shape)

    visualize(frames)
