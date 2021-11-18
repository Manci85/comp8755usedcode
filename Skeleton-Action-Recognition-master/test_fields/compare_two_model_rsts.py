import pickle
import numpy as np
from tqdm import tqdm
import torch

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


def open_result_file(a_path):
    with open(a_path, 'rb') as r_path:
        r_path = list(pickle.load(r_path).items())
    return r_path


def obtain_label_paths(data_type):
    label_path_ntu120_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'
    label_path_ntu120_xset = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_label.pkl'
    label_path_ntu_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl'
    label_path_ntu_xview = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xview/val_label.pkl'
    label_path_kinetics = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/kinetics/val_label.pkl'

    if data_type == 'ntu120_xsub':
        the_label = label_path_ntu120_xsub
    elif data_type == 'ntu120_xset':
        the_label = label_path_ntu120_xset
    elif data_type == 'kinetics':
        the_label = label_path_kinetics
    elif data_type == 'ntu_xsub':
        the_label = label_path_ntu_xsub
    elif data_type == 'ntu_xview':
        the_label = label_path_ntu_xview
    else:
        raise NotImplementedError
    return the_label


if __name__ == '__main__':
    data_type = 'ntu_xsub'
    the_label = obtain_label_paths(data_type)

    with open(the_label, 'rb') as label:
        label = np.array(pickle.load(label))

    ## Baselines
    ntu120_xsub_jnt_1ht = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-07-29-ntu_xsub/2021-07-29T12-10-38/epoch57_test_score.pkl'

    ## DCT
    ntu120_xsub_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-08-01-ntu_xsub_jnt_tte_linear_8_w_orig/2021-08-01T23-05-12/epoch39_test_score.pkl'

    ## Nerf
    ntu_xset_jnt_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-07-29-ntu_xsub_jnt_ste_4_w_orig/2021-07-31T16-34-04/epoch57_test_score.pkl'

    path_1 = ntu120_xsub_jnt_1ht
    path_2 = ntu120_xsub_jnt_dct
    # path_2 = ntu_xset_jnt_nerf

    rst_1 = open_result_file(path_1)
    rst_2 = open_result_file(path_2)

    tgt_i_list = []
    for i in range(len(label[0])):
        rgb_idx, l = label[:, i]
        gt_l = int(l)

        _, ntu120_xsub_jnt_1ht_ = rst_1[i]
        _, ntu120_xsub_jnt_dct_ = rst_2[i]
        softmax_1 = torch.softmax(torch.tensor(ntu120_xsub_jnt_1ht_), dim=-1)
        softmax_2 = torch.softmax(torch.tensor(ntu120_xsub_jnt_dct_), dim=-1)
        pred_1 = np.argmax(ntu120_xsub_jnt_1ht_)
        pred_2 = np.argmax(ntu120_xsub_jnt_dct_)
        if pred_1 != gt_l and pred_2 == gt_l:
        # if pred_1 != gt_l and pred_2 != gt_l:
        #     if softmax_1[gt_l].item() < 0.1:
        #     if softmax_1[gt_l].item() < 0.1 and softmax_2[gt_l].item() >= 0.8:
            if (softmax_1[gt_l].item() > 0.4 and softmax_1[gt_l].item() < 0.5) and \
                    (softmax_2[gt_l].item() >= 0.9):
                print('Actual: ', actions[gt_l+1],
                      '\tPredicted 1:', actions[pred_1+1],
                      '\tPredicted 2:', actions[pred_2+1],
                      '\tsoftmax 1: ', '{:.2f}'.format(softmax_1[gt_l].item()),
                      '\tsoftmax 2: ', '{:.2f}'.format(softmax_2[gt_l].item()),
                      '\ti: ', i, 'rgb_idx: ', rgb_idx[-29:-9])
                tgt_i_list.append(i)

    print('tgt_i_list: \n', tgt_i_list)