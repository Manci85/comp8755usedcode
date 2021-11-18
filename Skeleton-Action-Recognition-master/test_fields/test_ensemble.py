import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import pickle
import numpy as np
from tqdm import tqdm
import torch

from test_fields.net_over_ensemble import NetOverEnsemble


def get_out_file(out_path):
    with open(out_path, 'rb') as tr_out:
        tr_out = list(pickle.load(tr_out).items())
    return tr_out


def get_training_data(tr_outs, label_path):
    with open(label_path, 'rb') as tr_label:
        tr_label = np.array(pickle.load(tr_label))

    the_labels = []
    for i in tqdm(range(len(tr_label[0]))):
        _, l = tr_label[:, i]
        the_labels.append(int(l))
    the_labels = torch.tensor(the_labels).cuda()

    out_lists = []
    for a_tr_out in tr_outs:
        out_list = []
        right_num = total_num = right_num_5 = 0
        for i in tqdm(range(len(tr_label[0]))):
            _, l = tr_label[:, i]
            _, tr_out_0_ = a_tr_out[i]
            r = tr_out_0_
            out_list.append(r)

            rank_5 = r.argsort()[-5:]

            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))

            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

        print('Top1 Acc: {:.4f}%'.format(acc * 100))
        print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

        out_list = np.array(out_list)
        out_lists.append(out_list)
        print('out list: ', out_list.shape)
    out_lists = np.stack(out_lists, axis=1)
    out_lists = torch.tensor(out_lists).cuda()
    return out_lists, the_labels


if __name__ == '__main__':
    # 训练的参数
    epochs = 200
    criteria = torch.nn.CrossEntropyLoss()

    # 网络的构建
    net_over_enm = NetOverEnsemble(3, 120).cuda()
    # optim = torch.optim.SGD(net_over_enm.parameters(), lr=1e-3, momentum=0.9)
    optim = torch.optim.Adam(net_over_enm.parameters(), lr=1e-3)
    # net_over_enm.load_state_dict(torch.load('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/test_fields/analysis/net_over_enm.pt'))

    tr_label_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_label.pkl'
    te_label_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'

    tr_out_0 = get_out_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/eval/test_bone_only_sgcn_trans/2020-12-13T18-28-14/epoch1_test_score.pkl')
    tr_out_1 = get_out_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/eval/20-12-09-ntu120_xsub_joint_bone_angle_only_sgcn/2020-12-13T18-36-43/epoch1_test_score.pkl')
    tr_out_2 = get_out_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/eval/test_bone_only_sgcn_trans/2020-12-13T21-26-17/epoch1_test_score.pkl')

    tr_outs = [tr_out_0, tr_out_1, tr_out_2]
    out_lists, the_labels = get_training_data(tr_outs, tr_label_path)

    for a_epoch in range(epochs):
        a_net_out = net_over_enm(out_lists).squeeze()
        loss = criteria(a_net_out, the_labels)
        print('epoch: ', a_epoch, 'loss: ', loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    torch.save(net_over_enm.state_dict(), 'analysis/net_over_enm.pt')

    # test
    te_out_0 = get_out_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_100_50_25/2020-12-06T09-36-38/epoch55_test_score.pkl')
    te_out_1 = get_out_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_joint_bone_angle_only_sgcn/2020-12-11T17-30-05/epoch55_test_score.pkl')
    te_out_2 = get_out_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_5_5_5/2020-12-05T08-17-25/epoch50_test_score.pkl'
    )

    te_outs = [te_out_0, te_out_1, te_out_2]

    out_lists, the_labels = get_training_data(te_outs, te_label_path)

    orig_ensemble = torch.sum(out_lists, dim=1).cpu()
    orig_ensem_rst = torch.argmax(orig_ensemble, dim=-1).squeeze()
    pred_acc_orig = torch.sum((orig_ensem_rst == the_labels.cpu()).int()) / float(the_labels.shape[0])
    print('orig pred acc: ', pred_acc_orig.item())

    net_preds = net_over_enm(out_lists).cpu()
    predicted = torch.argmax(net_preds, dim=-1).squeeze()
    pred_acc = torch.sum((predicted == the_labels.cpu()).int()) / float(the_labels.shape[0])
    print('pred acc: ', pred_acc.item())

    print('Become better: ', pred_acc.item() > pred_acc_orig.item())
