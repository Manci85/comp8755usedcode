import numpy as np
import pickle
from tqdm import tqdm

from ensembling.utils import open_result_file
from ensembling.score_paths.labels import *
from ensembling.score_paths.scores import *


class Ensembler:
    def __init__(self):
        self.ens_list = [
            # '{}_joint_msgcn_chron_loss', '{}_bone_msgcn_chron_loss',
            # '{}_joint_msgcn_tte_chron_loss', '{}_bone_msgcn_tte_chron_loss'
            '{}_bone_msgcn_chron_loss', '{}_bone_msgcn_tte_chron_loss'
        ]
        self.data_type = 'ntu60_xview'

        self.score_paths = score_paths
        self.score_dict = {}
        for a_key in self.score_paths.keys():
            self.score_dict[a_key] = open_result_file(self.score_paths[a_key])

        self.get_label()

    def get_label(self):
        the_label = label_paths[self.data_type]
        with open(the_label, 'rb') as label:
            self.label = np.array(pickle.load(label))

    def evaluate_ensembling(self):
        right_num = total_num = right_num_5 = 0
        for inst_idx in tqdm(range(len(self.label[0]))):
            logit_sum = 0
            for a_ens_id in self.ens_list:
                the_ens_id = a_ens_id.format(self.data_type)
                _, logit = self.score_dict[the_ens_id][inst_idx]
                logit_sum += logit
            r = logit_sum
            rgb_idx, l = self.label[:, inst_idx]

            rank_1 = np.argmax(r)  # top-1
            right_num += int(rank_1 == int(l))

            rank_5 = r.argsort()[-5:]  # top-5
            right_num_5 += int(int(l) in rank_5)

            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        print('Top1 Acc: {:.4f}%'.format(acc * 100))
        print('Top5 Acc: {:.4f}%'.format(acc5 * 100))


if __name__ == '__main__':
    an_ensembler = Ensembler()
    an_ensembler.evaluate_ensembling()
