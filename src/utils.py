import json
import codecs
from os.path import join
import pickle
import os

def load_json(filePath):
    with codecs.open(join(filePath), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(path, file_name, data):
    file_path = path + file_name
    with open(file_path, 'w') as file:
        json.dump(data, file)

def get_files_by_extension(directory, extension='.json'):
    import os
    json_files_data = []

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            json_files_data.append(filename)

    return json_files_data


def save_as_pkl(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_from_pkl(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


from datetime import datetime

def get_today_ymd_compact():
    return datetime.now().strftime("%y%m%d")

