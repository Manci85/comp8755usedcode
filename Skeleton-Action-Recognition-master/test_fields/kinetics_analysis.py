import json

def get_kinetics_dict():
    rtn = {}
    json_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/Datasets/kinetics_raw/kinetics_train_label.json'
    with open(json_path, 'r') as myfile:
        data_json = json.load(myfile)

    for a_key, a_value in data_json.items():
        rtn[int(a_value['label_index']) + 1] = a_value['label']

    return rtn

if __name__ == '__main__':
    get_kinetics_dict()