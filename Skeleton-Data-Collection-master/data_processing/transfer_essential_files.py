import os
from shutil import copyfile


class GenerateMinimumFiles:
    def __init__(self):
        self.root_path = '/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Collected-Data/Collected-Data/'
        self.tgt_root_path = '/media/zhenyue-qin/Sklt-Data-1T/Collected-Skeleton-Data-Minimum/'
        self.tgt_rgb_root_path = os.path.join(self.tgt_root_path, 'RGB')
        self.tgt_depth_root_path = os.path.join(self.tgt_root_path, 'Depth')
        self.tgt_skeleton_root_path = os.path.join(self.tgt_root_path, 'Skeleton')
        self.tgt_raw_skeleton_root_path = os.path.join(self.tgt_root_path, 'Raw_Skeleton')

    def cp_relevant_files(self):
        count_ = 0
        for path, subdirs, files in os.walk(self.root_path):
            for name in files:
                path_name = os.path.join(path, name)
                a_tgt_f_name = None
                split_path = path_name.split(os.path.sep)
                a_video_name = split_path[-1]
                second_last_name = split_path[-2]
                if name.endswith('.avi'):
                    if a_video_name == 'a_video.avi' or a_video_name == 'depth.avi':
                        file_name = second_last_name + '.avi'
                        a_tgt_f_name = os.path.join(*split_path[:-2], file_name)
                    elif 'a_video_' in a_video_name and second_last_name[:-5] in a_video_name:
                        file_name = second_last_name + '.avi'
                        a_tgt_f_name = os.path.join(*split_path[:-2], file_name)
                    elif 'Azure-' in second_last_name and ('meta' not in second_last_name):
                        a_tgt_f_name = os.path.join(*split_path)
                    else:
                        print('Warning! This file is dodgy: ', path_name, '\n tgt f name: ', a_tgt_f_name)
                        print('second_last_name: ', second_last_name)
                        print('video name: ', a_video_name)
                    if 'RGB' in a_tgt_f_name:
                        a_tgt_f_name = os.path.join(self.tgt_rgb_root_path, *a_tgt_f_name.split(os.path.sep)[6:])
                    elif 'Depth' in a_tgt_f_name:
                        a_tgt_f_name = os.path.join(self.tgt_depth_root_path, *a_tgt_f_name.split(os.path.sep)[6:])
                # if False:
                #     pass
                elif name.endswith('.npy'):
                    to_save_dir = path_name.split(os.path.sep)[7:10]
                    a_tgt_f_name = os.path.join(self.tgt_skeleton_root_path, *to_save_dir, 'npy', name)
                    # print('a tgt f name: ', a_tgt_f_name)
                elif name.endswith('.mp4'):
                    to_save_dir = path_name.split(os.path.sep)[7:10]
                    a_tgt_f_name = os.path.join(self.tgt_skeleton_root_path, *to_save_dir, 'animation', name)
                    # print('a tgt f name: ', a_tgt_f_name)
                elif name.endswith('.txt'):
                    to_save_dir = path_name.split(os.path.sep)[7:]
                    a_tgt_f_name = os.path.join(self.tgt_raw_skeleton_root_path, *to_save_dir)
                if a_tgt_f_name is not None:
                    count_ += 1
                    if count_ % 1000 == 0:
                        print('processing dir: ', count_, a_tgt_f_name)
                    if os.path.exists(a_tgt_f_name):
                        if count_ % 1000 == 0:
                            print('continued')
                        continue
                    # print('src: ', path_name)
                    new_dir = os.path.join(*os.path.split(a_tgt_f_name)[:-1])
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    assert 'Backup' not in a_tgt_f_name
                    assert a_tgt_f_name[0] == '/'
                    if count_ % 1000 == 0:
                        print(path_name, 'to \nnew: a tgt f name: ', a_tgt_f_name)
                    copyfile(path_name, a_tgt_f_name)


if __name__ == '__main__':
    gen_min_files = GenerateMinimumFiles()
    gen_min_files.cp_relevant_files()
