import os
import json
import csv
import pickle as pkl
import numpy as np
import pandas as pd


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


def load_user_coor(dataset):
    df_train = pd.read_csv('../../dataset/{}/{}_user_loc.csv'.format(dataset, 'train'), sep=',', encoding='utf-8',
                           names=['lat', 'lon'], quoting=csv.QUOTE_NONE)
    df_val = pd.read_csv('../../dataset/{}/{}_user_loc.csv'.format(dataset, 'valid'), sep=',', encoding='utf-8',
                         names=['lat', 'lon'], quoting=csv.QUOTE_NONE)
    df_test = pd.read_csv('../../dataset/{}/{}_user_loc.csv'.format(dataset, 'test'), sep=',', encoding='utf-8',
                          names=['lat', 'lon'], quoting=csv.QUOTE_NONE)

    # merge df
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    return df, len(df_train), len(df_val), len(df_test)


def load_loc_label(dataset):
    file_path = '../../dataset/{}/user_loc.csv'.format(dataset)
    loc_list = []
    with open(file_path, 'r', encoding='utf-8') as cf:
        lines = csv.reader(cf)
        for line in lines:
            # print(line)
            loc_list += [int(i) for i in line]
    return loc_list

def load_loc_coor(dataset):
    lat_path = '../../dataset/{}/classLatMedian.json'.format(dataset)
    lon_path = '../../dataset/{}/classLonMedian.json'.format(dataset)
    with open(lat_path, 'r', encoding='utf-8') as lat_file:
        classLat_tmp = json.load(lat_file)

    with open(lon_path, 'r', encoding='utf-8') as lon_file:
        classLon_tmp = json.load(lon_file)

    classLatMedian, classLonMedian = {}, {}
    for key in classLat_tmp:
        classLatMedian[int(key)] = float(classLat_tmp[key])
        classLonMedian[int(key)] = float(classLon_tmp[key])

    return classLatMedian, classLonMedian