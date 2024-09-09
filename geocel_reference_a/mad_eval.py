import csv
import json
import pandas as pd
from haversine import haversine
import numpy as np


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


def load_user_coor(dataset):
    
    df_test = pd.read_csv('../../dataset/{}/{}_user_loc.csv'.format(dataset, 'test'), sep=',', encoding='utf-8',
                          names=['lat', 'lon'], quoting=csv.QUOTE_NONE)

    return df_test


def eval(pred_labels, true_labels, user_true_coor, class_lat, class_lon):
    assert len(pred_labels) == len(true_labels), "Number of predictions and true labels must be the same"

    err_distances = [haversine((class_lat[pred_labels[i]], class_lon[pred_labels[i]]),
                               user_true_coor[i]) for i in range(len(pred_labels))]
    accuracy = np.mean([1 if pred_labels[i] == true_labels[i] else 0 for i in range(len(pred_labels))])

    acc_161 = len([dist for dist in err_distances if dist <= 161]) / len(err_distances)
    mean_ed = np.mean(err_distances)
    median_ed = np.median(err_distances)
    return err_distances, accuracy, acc_161, mean_ed, median_ed



if __name__ == '__main__':
    dataset = 'cmu'
    loc_list = load_loc_label(dataset)
    user_coor_list = load_user_coor(dataset).values.tolist()
    class_lat, class_lon = load_loc_coor(dataset)
    user_pred_loc = []
    with open('./geocel_result_0.9_0.15_cover.txt'.format(dataset), 'r', encoding='utf-8') as cf:
        lines = csv.reader(cf, delimiter='\t')
        i = 0
        for line in lines:
            if i < 5685+1895:
                i += 1
                continue
            elif i < 9475:
                i += 1
                user_pred_loc += [int(line[1])]
    print(len(user_pred_loc))
    err_dists, accuracy, acc_161, mean_ed, median_ed = eval(user_pred_loc, loc_list[5685+1895:9475], user_coor_list, class_lat, class_lon)
    print('Accuracy: {:.4f}, Acc@161: {:.4f}, Mean ED: {:.4f}, Median ED: {:.4f}'.format(accuracy, acc_161, mean_ed, median_ed))
    # save error distances as csv file
    with open('./geocel_err_dist_0.9_0.15_cover.csv'.format(dataset), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(err_dists)