import pickle


def open_result_file(a_path):
    with open(a_path, 'rb') as r_path:
        r_path = list(pickle.load(r_path).items())
    return r_path