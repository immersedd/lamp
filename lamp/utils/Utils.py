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
    # Saving the list as JSON
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


import os


def delete_sql_files(file_list, folder_path, keep=False):

    deleted = 0
    file_list_set = set(file_list)

    for fname in os.listdir(folder_path):
        if not fname.endswith(".sql"):
            continue
        full_path = os.path.join(folder_path, fname)

        if (not keep and fname in file_list_set) or (keep and fname not in file_list_set):
            os.remove(full_path)
            deleted += 1

