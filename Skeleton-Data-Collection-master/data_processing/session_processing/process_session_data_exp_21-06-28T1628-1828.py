from time import sleep

import skeleton_visualizer
import rgb_segmenter
import multiprocessing as mp
import os


def generate_random_string(a_len):
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(a_len))


class SkeletonSessionDataProcessor:
    def __init__(self, time_logger_csv_path, skeleton_file_path, rgb_file_path,
                 save_root):
        self.time_logger_csv_path = time_logger_csv_path
        self.skeleton_file_path = skeleton_file_path
        self.rgb_file_path = os.path.join(rgb_file_path, '*.png')
        self.depth_file_path = self.rgb_file_path.replace('RGB', 'Depth')

        a_random_str = generate_random_string(4)
        self.base_save_root = save_root + f'_{a_random_str}'
        self.skeleton_save_root = self.base_save_root.replace('BASE', 'Skeleton')
        self.rgb_save_root = self.base_save_root.replace('BASE', 'RGB')
        self.depth_save_root = self.base_save_root.replace('BASE', 'Depth')

        # check integrity
        assert save_root[3:13] in self.time_logger_csv_path and \
               save_root[3:13] in self.skeleton_file_path and \
               save_root[3:13] in self.rgb_file_path

        self.process()

    def process(self):
        pool = mp.Pool(3)

        # skeleton
        # skeleton_visualizer.get_action_clips(
        #     self.time_logger_csv_path, self.skeleton_file_path, self.skeleton_save_root
        # )
        pool.apply_async(skeleton_visualizer.get_action_clips, args=(
            self.time_logger_csv_path, self.skeleton_file_path, self.skeleton_save_root))

        # RGB
        # rgb_segmenter.get_action_clips(
        #     self.time_logger_csv_path, self.rgb_file_path, self.rgb_save_root, 'RGB'
        # )
        pool.apply_async(rgb_segmenter.get_action_clips, args=(
            self.time_logger_csv_path, self.rgb_file_path, self.rgb_save_root, 'RGB'))

        # Depth
        # rgb_segmenter.get_action_clips(
        #     self.time_logger_csv_path, self.depth_file_path, self.depth_save_root, 'Depth'
        # )
        pool.apply_async(rgb_segmenter.get_action_clips, args=(
            self.time_logger_csv_path, self.depth_file_path, self.depth_save_root, 'Depth'))

        pool.close()
        pool.join()


if __name__ == '__main__':

    a_time_logger = '/media/zhenyue-qin/Seagate Expansion Drive2/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/time_logger/saved_csv/2021-06-28T16-32-50_action_log.csv'
    a_save_root = f'../2021-06-28T16:30-18:30-Azure-'

    # Azure 1
    a_skeleton_path_1 = '/media/zhenyue-qin/Skeleton-1/Azure-1-Data/skeletonData-2021-06-28--16-29-27.txt'
    a_rgb_path_1 = a_skeleton_path_1.replace('skeletonData-', 'RGBImages').replace('.txt', '')
    a_save_root_1 = a_save_root + '1-BASE'

    # Azure 2
    # a_skeleton_path_2 = '/media/zhenyue-qin/Seagate Expansion Drive/Azure-2-Data/skeletonData-2021-06-28--16-28-24.txt'
    # a_rgb_path_2 = a_skeleton_path_2.replace('skeletonData-', 'RGBImages').replace('.txt', '')
    # a_save_root_2 = a_save_root + '2-BASE'

    # Azure 3
    # a_skeleton_path_3 = '/media/zhenyue-qin/Seagate Expansion Drive/Azure-3-Data/skeletonData-2021-06-28--16-28-20.txt'
    # a_rgb_path_3 = '/media/zhenyue-qin/Seagate Expansion Drive/Azure-3-Data/RGBImages2021-06-28--16-28-20'
    # a_save_root_3 = a_save_root + '3-BASE'

    # Azure 4
    # a_skeleton_path_4 = '/media/zhenyue-qin/HCC-Drive-4/Azure-4-Data/skeletonData-2021-06-28--16-28-49.txt'
    # a_rgb_path_4 = '/media/zhenyue-qin/HCC-Drive-4/Azure-4-Data/RGBImages2021-06-28--16-28-49'
    # a_save_root_4 = a_save_root + '4-BASE'

    # Azure 5
    # a_skeleton_path_5 = '/media/zhenyue-qin/HCC-Drive-5/Azure-5-Data/skeletonData-2021-06-28--16-28-09.txt'
    # a_rgb_path_5 = '/media/zhenyue-qin/HCC-Drive-5/Azure-5-Data/RGBImages2021-06-28--16-28-09'
    # a_save_root_5 = a_save_root + '5-BASE'

    a_skeleton_path_to_use = a_skeleton_path_1
    a_rgb_path_to_use = a_rgb_path_1
    a_save_root_to_use = a_save_root_1

    # Azure integrity check
    azure_id = a_save_root_to_use[-12:-5]
    assert (azure_id in a_skeleton_path_to_use and azure_id in a_rgb_path_to_use) or \
        (azure_id.replace('-', '') in a_skeleton_path_to_use and azure_id.replace('-', '') in a_rgb_path_to_use)

    sklt_sess_data_process = SkeletonSessionDataProcessor(a_time_logger, a_skeleton_path_to_use, a_rgb_path_to_use,
                                                          a_save_root_to_use)
