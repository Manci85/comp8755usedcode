import os
import glob
import shutil
from collections import defaultdict
import cv2

from hyperparameters import start_off, end_off
from skeleton_visualizer import get_all_start_end_time, convert_time_to_second, generate_random_string


def generate_a_video(image_folder, video_name, save_path=None):
    if save_path is None:
        video_name = os.path.join(image_folder, video_name)
    else:
        video_name = os.path.join(save_path, video_name)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fps = 18
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    remaining_img_num = len(images) % fps
    # images = images[0:-remaining_img_num]
    print('image folder: ', image_folder, 'len of images: ', len(images))
    if len(images) >= 500:
        return
    assert len(images) < 500

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def get_key_time_dict(rgb_img_paths):
    rtn_dict = {}
    for e in rgb_img_paths:
        a_f_name = os.path.split(e)[-1]
        split_e = a_f_name.replace('.png', '').split('-')
        a_time = int(split_e[1:][0]) * 3600 + int(split_e[1:][1]) * 60 + int(split_e[1:][2]) + \
            float(split_e[1:][3]) / 10000.0
        rtn_dict[a_time] = e
    return rtn_dict


def get_action_clips(log_csv, rgb_path, a_save_root_path, a_video_type):
    start_offset = start_off
    end_offset = end_off

    rgb_img_paths = glob.glob(rgb_path)
    time_path_dict = get_key_time_dict(rgb_img_paths)
    time_keys = sorted(time_path_dict.keys())

    # if not os.path.exists(a_save_root_path):
    #     os.makedirs(a_save_root_path)
    # file name count dict
    file_name_count_dict = defaultdict(int)

    at_list = get_all_start_end_time(log_csv)
    prev_time = None

    at_idx = 0  # start line
    action_skeleton_dict = {}

    cur_at = at_list[at_idx]

    # Update actions
    cur_action = cur_at[0]
    cur_start_time = convert_time_to_second(cur_at[1]) + start_offset
    cur_end_time = convert_time_to_second(cur_at[2]) + end_offset
    forward_backward = cur_at[3]
    active_passive = cur_at[4]
    action_skeleton_dict[cur_action] = {}

    is_acting = False

    count_ = 0
    for the_time in time_keys:
        # if count_ % 1 == 0:
        #     print('prev_time: ', prev_time, 'the_time', the_time,'cur_start_time: ', cur_start_time,
        #           '')
        # count_ += 1
        if prev_time is None:
            prev_time = the_time
            continue
        if (prev_time <= cur_start_time and the_time >= cur_start_time) or \
                ((prev_time > cur_start_time and the_time > cur_start_time) and not is_acting):
            relevant_paths = []
            is_acting = True

        if is_acting:
            relevant_paths.append(time_path_dict[the_time])

        if (prev_time < cur_end_time and the_time >= cur_end_time) or \
                ((prev_time > cur_end_time and the_time > cur_end_time) and is_acting):
            is_acting = False

            a_file_name = '{}_{}_{}'.format(
                cur_action.replace(': ', '-').replace(' ', '-'), forward_backward, active_passive
            )
            file_name_count = file_name_count_dict[a_file_name]
            to_save_path = os.path.join(a_save_root_path + '_meta', a_file_name + '-' + str(file_name_count)).replace(':', 'h')
            vid_to_save_path = os.path.join(a_save_root_path + '_videos').replace(':', 'h')

            # Update actions
            at_idx += 1
            try:
                cur_at = at_list[at_idx]
            except IndexError:
                print('Got index error in RGB at: ', at_idx)
                continue

            if not os.path.exists(vid_to_save_path):
                os.makedirs(vid_to_save_path)

            if not os.path.exists(to_save_path):
                os.makedirs(to_save_path)
                for a_file in relevant_paths:
                    shutil.copy2(a_file, os.path.join(to_save_path, os.path.split(a_file)[-1]))
            generate_a_video(to_save_path, 'a_video_{}_-_{}_{}.avi'.format(a_file_name, str(file_name_count),
                                                                           a_video_type),
                             save_path=vid_to_save_path)
            file_name_count_dict[a_file_name] += 1

            print(a_video_type, 'a file name: ', a_file_name)
            # assert 0

            cur_action = cur_at[0]
            cur_start_time = convert_time_to_second(cur_at[1]) + start_offset
            cur_end_time = convert_time_to_second(cur_at[2]) + end_offset

            forward_backward = cur_at[3]
            active_passive = cur_at[4]
            action_skeleton_dict[cur_action] = {}

        prev_time = the_time


if __name__ == '__main__':
    # 修改这里
    # video_type = 'RGB'
    video_type = 'Depth'
    # {video_type}

    log_csv = '/media/zhenyue-qin/Elements/Data-Collection/Skeleton-Data-Collection/time_logger/saved_csv/2021-05-30T14-40-38_action_log.csv'
    rgb_dir = f'/media/zhenyue-qin/One Touch1/Azure1-Data/{video_type}Images2021-05-30--14-38-34/*.png'  # 修改这里
    a_random_str = generate_random_string(4)
    save_root_dir = f'../2021-05-30-14:30-16:30-Azure-1-{video_type}-{a_random_str}'  #修改这里
    assert save_root_dir[3:13] in rgb_dir and save_root_dir[3:13] in log_csv
    get_action_clips(log_csv, rgb_dir, save_root_dir, video_type)
