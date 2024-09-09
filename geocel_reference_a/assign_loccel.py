from haversine import haversine
import csv
import json


def load_geoloc(dataset, city=None):
    
    lat_path = '../../dataset/{}/classLatMedian.json'.format(dataset)
    lon_path = '../../dataset/{}/classLonMedian.json'.format(dataset)
    with open(lat_path, 'r', encoding='utf-8') as lat_file:
        classLatMedian = json.load(lat_file)
    
    with open(lon_path, 'r', encoding='utf-8') as lon_file:
        classLonMedian = json.load(lon_file)

    return classLatMedian, classLonMedian


def load_cel_coor(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=',')
        cel_coor = []
        for line in lines:
            cel_coor.append((float(line[0]), float(line[1])))
    return cel_coor


if __name__ == '__main__':
    dataset = 'cmu'
    loccel_geo_file = './loc_cel_cmu_geo.csv'
    classLatMedian, classLonMedian = load_geoloc(dataset)
    cel_coor = load_cel_coor(loccel_geo_file)
    loccel_label = []
    for coor in cel_coor:
        # print(coor)
        min_dist = 100000
        min_class = None
        for class_name, lat in classLatMedian.items():
            lon = float(classLonMedian[class_name])
            lat = float(lat)
            dist = haversine(coor, (lat, lon))
            if dist < min_dist:
                min_dist = dist
                min_class = class_name
        # print(min_dist)
        loccel_label.append(min_class)
    # print(loccel_label)

    with open(f'../../dataset/{dataset}/seed_geocel.txt', 'w', encoding='utf-8') as f:
        for i, label in enumerate(loccel_label):
            idx = 9475 + i
            f.write(str(idx) + '\t' + str(label) + '\t' + '0.6' + '\n')