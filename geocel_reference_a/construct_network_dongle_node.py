# follow rahimi
# we add text LR result as dongle node in network
# link infer result and valid/test nodes

import networkx as  nx
import numpy as np


def add_dongle_node(dongle_node_list,link_nodes, g, nodes_num):
    for i in range(len(dongle_node_list)):
        new_node = i + nodes_num
        g.add_node(new_node)
        g.add_edge(new_node, link_nodes[i])
    return g


def construct_network_dongle_node(infer_result_valid, infer_result_test, valid_nodes, test_nodes, network):
    node_num = len(network.nodes)
    network = add_dongle_node(infer_result_valid, valid_nodes, network, node_num)
    node_num = len(network.nodes)
    network = add_dongle_node(infer_result_test, test_nodes, network, node_num)

    # reconstruct label matrix
    # label_matrix = np.vstack((infer_result_valid, infer_result_test))
    print('network nodes: ', len(network.nodes))
    return network


if __name__ == '__main__':
    # test functions
    import pickle as pkl
    g_mat = pkl.load(open('../../dataset/cmu/graph_mat_dbscan.cmu.pkl', 'rb'))
    g = nx.from_scipy_sparse_array(g_mat)

    # load valid & test labels from LR
    infer_result_valid = np.load('../result/LR_cmu_valid.npy')
    infer_result_test = np.load('../result/LR_cmu_test.npy')
    print(infer_result_test.shape)
    # print(len(infer_result_test))
    # valid & test nodes
    train_slice, valid_slice, test_slice = 5685, 5685+1895, 9475
    valid_nodes = [i for i in range(train_slice, valid_slice)]
    test_nodes = [i for i in range(valid_slice, 9475)]
    g_new = construct_network_dongle_node(infer_result_valid, infer_result_test, valid_nodes, test_nodes, g)

    # # save g_new as sparse matrix
    with open('../../dataset/cmu/graph_mat_dbscan_lr.pkl', 'wb') as f:
        pkl.dump(nx.to_scipy_sparse_array(g_new), f)