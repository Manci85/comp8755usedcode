import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import re


# NTU RGB+D 60/120 Action Classes
actions = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

ntu_skeleton_bone_pairs = tuple((i - 1, j - 1) for (i, j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
))

ntu_tgt_angle = tuple((i-1, j-1) for (i, j) in (
    (11, 10),
    (10, 9),
    (5, 6),
    (6, 7)
))
# ntu_tgt_angle = ()

bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,

    # NTU general
    'ntu': ntu_skeleton_bone_pairs,
}


def process_data(process_type, a_data, a_label):
    rtn_data = []
    rtn_label_0 = []
    rtn_label_1 = []
    if process_type == 'single_person':
        data_idx = 0
        for a_data in a_data:
            for ppl_id in range(a_data.shape[-1]):
                a_ppl = a_data[:, :, :, ppl_id]
                if np.max(a_ppl) > 0.01:
                    rtn_data.append(np.expand_dims(a_ppl, axis=-1))
                    rtn_label_0.append(a_label[0][data_idx])
                    rtn_label_1.append(a_label[1][data_idx])
            data_idx += 1
    else:
        raise NotImplementedError
    rtn_data = np.stack(rtn_data, axis=0)
    return rtn_data, (rtn_label_0, rtn_label_1)


def visualize(args):
    plotted_num = 0
    processed_classes = defaultdict(int)
    process_num_per_cls = 1000

    data_path = args.datapath or './data/{}/val_data_joint.npy'.format(args.dataset)
    label_path = args.labelpath or './data/{}/val_label.pkl'.format(args.dataset)

    data = np.load(data_path, mmap_mode='r')
    with open(label_path, 'rb') as f:
        labels = pickle.load(f, encoding='latin1')

    # Data modification
    # data, labels = process_data('single_person', data, labels)

    bones = bone_pairs[args.dataset]
    print(f'Dataset: {args.dataset}\n')

    def animate(skeletons):
        # 这里skeleton是3*25, 3是3d坐标, 25是joint的数量.
        ax.clear()
        # ax.set_xlim([-0.4, 0.6])
        # ax.set_xlim([-1.0, 1.0])
        # ax.set_xlim([-0.3, 0.7])

        # ax.set_ylim([-1.0, 1.0])
        # ax.set_ylim([-0.5, 0.5])

        # ax.set_zlim([-1.0, 1.0])
        # ax.set_zlim([-0.5, 0.5])

        # ax.set_xlim([-0.3, 0.7])
        # ax.set_ylim([-0.7, 0.1])
        # ax.set_zlim([-0.5, 0.7])

        ax.set_xlim([-0.2, 0.8])
        ax.set_ylim([-0.75, -0.15])
        ax.set_zlim([-0.5, 0.7])

        # ax.set_xlim([-0.2, 0.8])
        # ax.set_ylim([-0.7, -0.1])
        # ax.set_zlim([-0.5, 0.9])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # person 1
        k = 0
        color_list = ('blue', 'orange', 'cyan', 'purple')
        color_idx = 0
        while k < skeletons.shape[0]:
            for i, j in bones:
                if (i, j) in ntu_tgt_angle or (j, i) in ntu_tgt_angle:
                    continue
                joint_locs = skeletons[:, [i, j]]
                # plot them
                ax.plot(joint_locs[k+0], joint_locs[k+1], joint_locs[k+2], color=color_list[color_idx])

            for i, j in ntu_tgt_angle:
                joint_locs = skeletons[:, [i, j]]
                # plot them
                ax.plot(joint_locs[k+0], joint_locs[k+1], joint_locs[k+2], color='red', linestyle='dashed')
            k += 3
            color_idx = (color_idx + 1) % len(color_list)

        action_class = labels[1][index] + 1
        action_name = actions[action_class]
        pred_idx_exist = "pred_idx" in locals()
        rgb_id = os.path.split(labels[0][index])[-1].replace('.skeleton', '')

        if pred_idx_exist:
            pred_class = pred_idx + 1
            pred_name = actions[pred_class]
            title = '\nSkeleton {} Frame #{} of 300 from {}\n' \
                    '(Action {}: {}) (Pred {}: {}) \n(RGB ID: {})'.format(index, skeleton_index[0], args.dataset,
                                                                          action_class, action_name, pred_class, pred_name,
                                                                          rgb_id)
        else:
            title = '\nSkeleton {} Frame #{} of 300 from {}\n ' \
                    '(Action {}: {}) \n(RGB ID: {})'.format(index, skeleton_index[0], args.dataset,
                                                            action_class, action_name,
                                                            rgb_id)
        plt.title(title)
        skeleton_index[0] += 1
        return ax

    for an_entry in args.indices:

        if isinstance(an_entry, tuple) and len(an_entry) == 2:
            index = int(an_entry[0])
            pred_idx = int(an_entry[1])
        else:
            index = an_entry
        # get data
        try:
            skeletons = np.copy(data[index])
        except:
            print('final instance num: ', plotted_num)
            assert 0

        action_class = labels[1][index] + 1
        action_name = actions[action_class].replace('/', '-').replace('_', '-')

        if 'pred_idx' in locals():
            pred_class = pred_idx + 1
            pred_name = actions[pred_class].replace('/', '-').replace('_', '-')

        # kicking
        # if processed_classes[action_class] > process_num_per_cls or action_class != 51:
        # Mutual actions
        if processed_classes[action_class] > process_num_per_cls \
                or (
                False \
                # (action_class != 11)
                ):
            continue

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])
        processed_classes[action_class] += 1

        save_name = '{}_{}'.format(action_class, action_name)

        # print(f'Sample index: {index}\nAction: {action_class}-{action_name}\n')  # (C,T,V,M)

        # Pick the first body to visualize
        skeleton1 = skeletons[..., 0]  # out (C,T,V)
        # make it shorter
        shorter_frame_start = 0
        shorter_frame_end = 300
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

        # More than one person
        if skeleton_frames_2 is not None and np.max(skeleton_frames_2) > 0.01:
            skeleton_frames_1[:, 0, :] += 0
            skeleton_frames_2[:, 0, :] += 0
            ani = FuncAnimation(fig, animate, np.concatenate((skeleton_frames_1, skeleton_frames_2),
                                                             axis=1))
        else:
            ani = FuncAnimation(fig, animate, skeleton_frames_1)

        plt.title('Skeleton {} from {} test data Action {}'.format(index, args.dataset, action_class))
        # plt.show()
        ani_path = os.path.join('visuals/orig_wrong_dct_right', args.dataset, save_name)
        if not os.path.exists(ani_path):
            os.makedirs(ani_path)
        rgb_id = os.path.split(labels[0][index])[-1].replace('.skeleton', '')
        # if rgb_id != 'S006C001P007R002A053':
        #     print('continue rgb id: ', rgb_id)
        #     continue
        # else:
        #     print('caught one! ')
        if "pred_idx" in locals():
            file_name = os.path.join(ani_path, 'skeleton_{}_action_{}_{}_pred_{}_{}_{}.mp4'.format(
                index, action_class, action_name, pred_class, pred_name, rgb_id))

        else:
            file_name = os.path.join(ani_path, 'skeleton_{}_action_{}_{}_{}.mp4'.format(
                index, action_class, action_name, rgb_id))
        # ani.save(file_name, dpi=200, writer='ffmpeg')
        try:
            ani.save(file_name, dpi=200, writer='ffmpeg')
            if (plotted_num % 1 == 0):
                print('plot num: ', plotted_num, 'an_entry: ', an_entry,
                      'action: ', action_name)
            plotted_num += 1
        except:
            print('One did not save: ', save_name)
            print('final plotted num: ', plotted_num)
            assert 0
            # continue


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
    # NOTE:Only supports joint data
    parser = argparse.ArgumentParser(description='NTU RGB+D Skeleton Visualizer')

    parser.add_argument('-d', '--dataset',
                        choices=['ntu/xview', 'ntu/xsub', 'ntu120/xset', 'ntu120/xsub'])
    parser.add_argument('-p', '--datapath',
                        help='location of dataset numpy file')
    parser.add_argument('-l', '--labelpath',
                        help='location of label pickle file')
    parser.add_argument('-i', '--indices',
                        type=int,
                        nargs='+',
                        help='the indices of the samples to visualize')

    args = parser.parse_args()

    # args.dataset = 'ntu120/xsub'
    args.dataset = 'ntu/xsub'

    # idx, preds = get_wrong_sample_idx(
    #     '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-27-ntu_xsub_jnt_1ht_hyp/2021-01-28T09-44-17/wrong_file.txt'
    # )
    # start_idx = 0
    # idx = idx
    # args.indices = zip(idx[start_idx:], preds[start_idx:])
    args.indices = [782, 935, 1020, 1471, 1476, 1558, 2042, 2468, 2549, 3003, 3383, 3563, 3723, 3850, 4035, 4580, 4634, 4868, 5512, 5531, 5743, 6686, 7021, 7222, 7401, 7757, 7765, 8518, 8691, 8701, 9138, 9963, 10088, 10092, 10360, 10703, 10706, 10716, 11294, 11544, 11939, 12432, 12519, 12899, 12998, 12999, 14368, 14420, 14699, 14794, 15128, 15270, 15360, 15391, 15873, 15899, 15960, 16232]
    # args.indices = [92, 220, 240, 568, 579, 639, 798, 902, 935, 938, 945, 1062, 1504, 1558, 1645, 1715, 2168, 2549, 2638, 2696, 2945, 3182, 3195, 3304, 3546, 3662, 3701, 3702, 3716, 4056, 4139, 4156, 4252, 4274, 4653, 4845, 4882, 5060, 5234, 5239, 5308, 5538, 5688, 5758, 5775, 5855, 5864, 6011, 6039, 6047, 6068, 6106, 6154, 6169, 6204, 6205, 6255, 6291, 6297, 6398, 6479, 6519, 6841, 7037, 7248, 7468, 7683, 7828, 7971, 8036, 8049, 8050, 8064, 8156, 8301, 8342, 8365, 8404, 8480, 8482, 8669, 8672, 8687, 8768, 8794, 8835, 8894, 8918, 8923, 8931, 8963, 9009, 9037, 9053, 9106, 9138, 9241, 9273, 9292, 9294, 9335, 9390, 9448, 9480, 9907, 9928, 9977, 10230, 10258, 10424, 10501, 10567, 10582, 10685, 10709, 10820, 11226, 11406, 11443, 11843, 11922, 11998, 12116, 12206, 12542, 12545, 12571, 12576, 12624, 12931, 13005, 13008, 13120, 13150, 13183, 13198, 13214, 13222, 13283, 13522, 13656, 13660, 13841, 13978, 14318, 14642, 14720, 14744, 14805, 14916, 14936, 14941, 15031, 15127, 15236, 15255, 15262, 15272, 15352, 15362, 15366, 15918, 15926, 16002, 16028, 16138, 16207, 16235, 16237, 16337, 16656, 16696, 16969, 17002, 17090, 17223, 17224, 17238, 17369, 17373, 17377, 17436, 17525, 17916, 17984, 18121, 18127, 18184, 18199, 18360, 18376, 18415, 18416, 18492, 18532, 18563, 18598, 18671, 18799, 18894, 18939, 19202, 19221, 19249, 19267, 19285, 19368, 19406, 19489, 19553, 19605, 19670, 19735, 19847, 19911, 19966, 19975, 20043, 20089, 20142, 20177, 20203, 20319, 20356, 20404, 20569, 20618, 20624, 20862, 20924, 20983, 21045, 21099, 21173, 21275, 21295, 21319, 21497, 21513, 21564, 21589, 21591, 21679, 21739, 21763, 21782, 21867, 21875, 21912, 21951, 21956, 21959, 21969, 21993, 22034, 22073, 22079, 22089, 22132, 22174, 22239, 22242, 22259, 22348, 22377, 22380, 22392, 22420, 22431, 22455, 22471, 22479, 22482, 22510, 22563, 22572, 22644, 22652, 22690, 22694, 22790, 22791, 22883, 22948, 23003, 23014, 23028, 23067, 23071, 23132, 23134, 23168, 23244, 23258, 23331, 23431, 23489, 23491, 23561, 23654, 23675, 23729, 23735, 23808, 23853, 23863, 23975, 24147, 24157, 24335, 24385, 24391, 24540, 24571, 24635, 24641, 24642, 24676, 24700, 24716, 24721, 24778, 24905, 24991, 25048, 25116, 25214, 25217, 25223, 25239, 25243, 25261, 25341, 25369, 25379, 25404, 25418, 25438, 25457, 25529, 25534, 25560, 25565, 25600, 25739, 25759, 25769, 25783, 25787, 25861, 25898, 26042, 26058, 26066, 26111, 26126, 26132, 26187, 26251, 26290, 26329, 26338, 26339, 26383, 26459, 26463, 26487, 26503, 26504, 26522, 26542, 26553, 26590, 26602, 26605, 26607, 26645, 26646, 26665, 26721, 26722, 26727, 26734, 26735, 26736, 26801, 26827, 26836, 26837, 26838, 26839, 26845, 26860, 26955, 26961, 26964, 26965, 26973, 26976, 26985, 26995, 27041, 27139, 27141, 27158, 27213, 27266, 27320, 27362, 27378, 27402, 27408, 27416, 27454, 27535, 27539, 27587, 27630, 27657, 27661, 27692, 27741, 27746, 27765, 27799, 27856, 27870, 27876, 27891, 27919, 27969, 28012, 28040, 28057, 28157, 28158, 28172, 28173, 28402, 28407, 28434, 28509, 28529, 28572, 28581, 28630, 28655, 28660, 28703, 28724, 28731, 28749, 28756, 28866, 28889, 28929, 28957, 28970, 28998, 29000, 29015, 29053, 29057, 29063, 29109, 29117, 29237, 29286, 29289, 29307, 29356, 29418, 29455, 29476, 29481, 29484, 29508, 29538, 29586, 29611, 29624, 29643, 29677, 29749, 29755, 29763, 29853, 29968, 30419, 30427, 30534, 30559, 30860, 30865, 30879, 31113, 31118, 31165, 31173, 31224, 31230, 31275, 31328, 31393, 31427, 31445, 31451, 31452, 31457, 31490, 31545, 31687, 31711, 31760, 31808, 31833, 31912, 31946, 31990, 32040, 32104, 32124, 32354, 32435, 32480, 32494, 32605, 32610, 32638, 32670, 32724, 32725, 32729, 32830, 32985, 33202, 33226, 33361, 33512, 33554, 33862, 33865, 33914, 33918, 33972, 33977, 34034, 34242, 34290, 34341, 34509, 34512, 34572, 34612, 34629, 34637, 34776, 34877, 34982, 35023, 35027, 35167, 35169, 35294, 35354, 35407, 35468, 35499, 35582, 35585, 35588, 35826, 35878, 35890, 35909, 36012, 36072, 36129, 36135, 36276, 36309, 36336, 36355, 36473, 36520, 36606, 36620, 36654, 36683, 36707, 36719, 36812, 36951, 36965, 37160, 37215, 37259, 37495, 37500, 37566, 37571, 37674, 37734, 37813, 37861, 37896, 37949, 37983, 38004, 38212, 38222, 38240, 38473, 38517, 38628, 38650, 38683, 38742, 38750, 38804, 38815, 38881, 38938, 38945, 39005, 39031, 39085, 39096, 39108, 39195, 39201, 39206, 39215, 39244, 39324, 39372, 39398, 39404, 39405, 39477, 39552, 39553, 39555, 39561, 39645, 39723, 39735, 39757, 39822, 39827, 39892, 39945, 39946, 40020, 40024, 40060, 40061, 40077, 40099, 40113, 40139, 40149, 40153, 40163, 40178, 40214, 40252, 40261, 40285, 40291, 40343, 40393, 40409, 40450, 40456, 40463, 40474, 40478, 40580, 40584, 40627, 40628, 40662, 40675, 40690, 40691, 40704, 40825, 40852, 40885, 40897, 40930, 40937, 41171, 41346, 41357, 41382, 41411, 41421, 41447, 41470, 41488, 41524, 41717, 41833, 41952, 41968, 42060, 42078, 42085, 42113, 42142, 42173, 42194, 42205, 42213, 42306, 42364, 42402, 42552, 42557, 42627, 42644, 42660, 42661, 42778, 42790, 42808, 42855, 42917, 42920, 42931, 43127, 43190, 43230, 43306, 43308, 43328, 43337, 43342, 43362, 43433, 43455, 43489, 43549, 43569, 43588, 43798, 43800, 43830, 43843, 43918, 43942, 43947, 44019, 44041, 44058, 44086, 44175, 44187, 44216, 44407, 44453, 44562, 44583, 44605, 44619, 44620, 44623, 44710, 44717, 44850, 44948, 44950, 44956, 45008, 45181, 45260, 45286, 45304, 45311, 45313, 45371, 45372, 45568, 45675, 45681, 45762, 45770, 45910, 45917, 45922, 45964, 46039, 46059, 46065, 46142, 46168, 46207, 46219, 46289, 46312, 46397, 46409, 46556, 46625, 46820, 46983, 46996, 47190, 47274, 47512, 47527, 47533, 47534, 47610, 47639, 47655, 47657, 47673, 47699, 47700, 47701, 47778, 47787, 47897, 48010, 48019, 48042, 48066, 48413, 48435, 48490, 48599, 48669, 48732, 48837, 48857, 48874, 49030, 49072, 49440, 49449, 49523, 49556, 49654, 50036, 50280, 50334, 50336, 50339, 50412, 50420, 50427, 50583, 50668, 50694, 50708, 50769, 50788, 50841, 50873]
    # args.indices = [5748]
    visualize(args)
