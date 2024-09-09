import networkx as nx
import re
import csv
import pandas as pd
import json
import pickle as pkl
from haversine import haversine
import numpy as np
from DBSCAN import *


def data_loader(file_path, encoding):
    df = pd.read_csv(file_path, sep='\t', encoding=encoding, names=['user', 'lat', 'lon', 'text'],
                     quoting=csv.QUOTE_NONE, error_bad_lines=False)
    return df


def calculate_haversine(data):
    dis_mat = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            dis = haversine(data[i], data[j])
            dis_mat.append(dis)
    return np.array(dis_mat)


# def loc_cel(df_train, users):
# #     print('train user num:', len(users))
#     if len(users) == 1:
#         return -1
# #     dpca = DensityPeakCluster(density_threshold=3, distance_threshold=5, anormal=False)
#     dis_mat = calculate_haversine(np.array(df_train.iloc[users, [1, 2]]))
#     if len(users) == 2:
#         if dis_mat[0] > 100:
#             return -1
#         else:
#             return 1
#     # print predict label
#     c_v = np.std(dis_mat)/np.mean(dis_mat)
#     if c_v > 0.5:
#         return -1
#     else:
#         return 1
#     return dpca.labels_

def loc_cel(df_train, users):
    if len(users) <= 1:
        return False, 0
    loc_list = np.array(df_train.iloc[users, [1, 2]])
    print(loc_list)
    loc_ = calc_aggreg_dbscan(loc_list, min_pt=0.3*len(users), eps=70)
    loc_geo = 0
    if loc_:
        loc_geo = np.mean(loc_list, axis=0)
        print(loc_geo)
    return loc_, loc_geo


def add_word_edges(g, g_word):
    edge_list = g_word.edges
    for edge in edge_list:
        if not g.has_edge(edge[0], edge[1]):
            g.add_edge(edge[0], edge[1])
    return g

def get_user_graph(dataset, df_user, df_train, celebrity_threshold=5):
    g = nx.Graph()
    origin_nodes = df_user['user'].tolist()

    origin_nodes_loc = origin_nodes
    node_id = {node: id for id, node in enumerate(origin_nodes_loc)}
    # record node_id
    # save_nodes
    # with open('../../dataset/{}/user_nodeid.json'.format(dataset), 'w', encoding='utf-8') as jf:
    #     json.dump(node_id, jf)

    g.add_nodes_from(node_id.values())
    pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    # pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    pattern1 = re.compile(pattern1)
    # pattern2 = re.compile(pattern2)

    print('adding edges and some out nodes')
    for i in range(len(df_user)):
        user = df_user.user[i]
        user_id = node_id[user]
        mentions = [m.lower() for m in pattern1.findall(df_user.text[i])]  # find mentions from ith user's text
        # topics = [t.lower() for t in pattern2.findall(df_user.text[i])]
        # TODO: maybe need mention times
        # mention words can be added here,mention times can be get
        # mention_times = {}
        # for mention in mentions:
        #     if mention not in mention_times:
        #         mention_times[mention] = mentions.count(mention)

        idmentions = set()  # userid for the mentioned user
        for m in mentions:
            if m in node_id:  #
                idmentions.add(node_id[m])
            else:  # generage new id for mentioned  user that not it the existing user list
                id = len(node_id)
                node_id[m] = id
                idmentions.add(id)
        if len(idmentions) > 0:  # add mentioned users into the graph
            g.add_nodes_from(idmentions)

        # add_edges
        # TODO: maybe weights
        for id in idmentions:
            if not g.has_edge(user_id, id):
                g.add_edge(user_id, id)
            else:
                ...
        # for t in topic_mentions:
        #     if not g.has_edge(user_id, t):
        #         g.add_edge(user_id, t)

    # delete celebrity
    celebrities = []
    loc_celebrities = set()
    cnt_cele = 0
    loc_celebrities_num = 0
    loccel_dict = {}
    for i in range(len(origin_nodes_loc), len(node_id)):
        deg = g.degree(i)
        # print('degree:', deg)
        if deg == 1:  #大于50的直接认为是全局名人
            celebrities.append(i)
        elif deg > celebrity_threshold:
            nbrs = g.neighbors(i)
            train_nbrs = [nbr for nbr in nbrs if nbr in range(len(df_train))]
            loc_cele, loccel_geo = loc_cel(df_train, train_nbrs)
#             print(loc_cele)
            if loc_cele:
                loc_celebrities_num += 1
                loc_celebrities.add(i)
                loccel_dict[i] = loccel_geo
                continue
            else:
                celebrities.append(i)
                cnt_cele += 1
    
    # 保存本地名人节点的位置为csv
    with open('./loc_cel_cmu_geo.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in loccel_dict.items():
            writer.writerow(list(value))

    g.remove_nodes_from(celebrities)
    print(
        'removing %d celebrity nodes with degree higher than %d' % (len(celebrities), celebrity_threshold))
    print('local celebrity nodes num: %d' % loc_celebrities_num)
    print('global celebrity nodes num: %d' % cnt_cele)
    # add undirect edges
    print("original nodes num: %d", len(origin_nodes))
    all_nodes = list(g.nodes())
    print("original nodes num: %d", len(all_nodes))
    new_delete_set = set()
    g_old = g.copy()
    for node in all_nodes:
        if node < len(origin_nodes):
            continue
        if node not in loc_celebrities:
            new_delete_set.add(node)
            nbrs = list(g_old.neighbors(node))
            for idx1 in range(len(nbrs)):
                nbr1 = nbrs[idx1]
                for idx2 in range(idx1 + 1, len(nbrs)):
                    nbr2 = nbrs[idx2]
                    if not g.has_edge(nbr1, nbr2):
                        g.add_edge(nbr1, nbr2)
    g.remove_nodes_from(new_delete_set)
    print("new nodes num: %d", len(g.nodes()))
    
    return g


def main(dataset):
    train_file = '../dataset/{}/user_info.train.csv'.format(dataset)
    valid_file = '../dataset/{}/user_info.valid.csv'.format(dataset)
    test_file = '../dataset/{}/user_info.test.csv'.format(dataset)

    df_train = data_loader(train_file, encoding='utf-8')
    df_valid = data_loader(valid_file, encoding='utf-8')
    df_test = data_loader(test_file, encoding='utf-8')

    df_tmp = pd.merge(df_train, df_valid, how='outer')
    df_all = pd.merge(df_tmp, df_test, how='outer')

    if len(df_all) == len(df_train) + len(df_valid) + len(df_test):
        g = get_user_graph(dataset, df_all, df_train, celebrity_threshold=5)
        # 使用pickle格式保存
        with open('../dataset/{}/graph_mat_dbscan.{}.pkl'.format(dataset, dataset), 'wb') as gf:
            pkl.dump(nx.to_scipy_sparse_array(g, format='csr'), gf)
    else:
        print('Bad!!')


if __name__ == '__main__':
    dataset = 'cmu'
    main(dataset)
