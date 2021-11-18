import os


def get_all_subdirs(a_path, tgt_str=None):
    subdirs= [os.path.join(a_path, o) for o in os.listdir(a_path) \
                    if os.path.isdir(os.path.join(a_path,o))]
    if tgt_str is None:
        return subdirs
    else:
        for a_subdir in subdirs:
            last_path_part = os.path.split(a_subdir)[-1]
            if tgt_str in last_path_part:
                return a_subdir
        return None