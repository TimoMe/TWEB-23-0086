import dgl

import pickle as pkl

graph_file = '../dataset/cmu/graph_mat_5.cmu.pkl'
graph_mat = pkl.load(open(graph_file, 'rb'))

g_dgl = dgl.from_scipy(graph_mat)
print('node number:', g_dgl.number_of_nodes())
graph_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_ids = [i for i in range(5685)]
dataloader = dgl.dataloading.DistNodeDataLoader(g_dgl, train_ids, graph_sampler,
                                                batch_size=4, shuffle=True)
input_nodes, output_nodes, blocks = next(iter(dataloader))
print(blocks)
