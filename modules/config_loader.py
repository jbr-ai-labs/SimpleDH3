import json


def load_config(prefix_path=''):
    with open(prefix_path + 'config.json') as json_data_file:
        data = json.load(json_data_file)
    return data
