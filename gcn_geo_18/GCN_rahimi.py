from eval import eval
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from dgl.nn.pytorch import GraphConv

import numpy as np
import argparse
from train import *


class HighGCN(nn.Module):
    def __init__(self, init_dim, d_model, class_num=129):
        super(HighGCN, self).__init__()
        #         self.fc = SparseFCLayer(in_features=w_onehot,
        #                                 out_features=d_model,
        #                                 bias=True)
        self.fc = nn.Linear(in_features=init_dim,
                            out_features=d_model,
                            bias=True)
        self.dropout = nn.Dropout(p=0.5)

        self.layernorm = nn.LayerNorm(d_model)
        activation = nn.LeakyReLU()
        self.activation = activation

        self.gconv1 = GraphConv(in_feats=d_model,
                                out_feats=d_model,
                                norm='both', weight=True, bias=False,
                                activation=activation,
                                allow_zero_in_degree=True)

        self.gconv2 = GraphConv(in_feats=d_model,
                                out_feats=d_model,
                                norm='both', weight=True, bias=True,
                                activation=activation,
                                allow_zero_in_degree=True)

        self.gconv3 = GraphConv(in_feats=d_model,
                                out_feats=class_num,
                                norm='both', weight=True, bias=True,
                                activation=activation,
                                allow_zero_in_degree=True)

        self.gate1 = nn.Linear(in_features=d_model,
                               out_features=1,
                               bias=True)
        self.gate2 = nn.Linear(in_features=d_model,
                               out_features=1,
                               bias=True)

    def forward(self, g, g_init_emb):
        # g_feat = self.dropout(torch.tanh(self.fc(g_init_emb)))
        # g_feat = self.layernorm(g_feat)
        g_feat = self.fc(g_init_emb)
        gat_emb0 = self.dropout(self.gconv1(g, g_feat))
        # gat_emb0 = self.layernorm(gat_emb0)

        t1 = (self.gate1(gat_emb0))
        gat_emb1 = gat_emb0 * t1 + g_feat*(1-t1)

        gat_emb = (self.gconv2(g, gat_emb1))
        t2 = self.dropout(self.gate2(gat_emb))
        gat_emb = gat_emb * t2 + gat_emb1 * (1 - t2)

        prob_mat = self.gconv3(g, gat_emb)

        return prob_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, choices=['cmu', 'twitter-us', 'twitter-world'], default='cmu')  # choose dataset

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-accumulate_steps', type=int, default=35)
    parser.add_argument('-b', '--batch_size', type=int, default=512)

    parser.add_argument('-d_model', type=int, default=300)

    parser.add_argument('-class_num', type=int, default=129)
    # parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-dropout', type=float, default=0.5)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_false')
    parser.add_argument('-device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    device = opt.device
    print('opt information:', opt)

    # load data
    print('loading data...')
    word_emb_train, word_emb_valid, word_emb_test = load_text_info(f'../dataset/{opt.dataset}',
                                                                                  opt.dataset)

    graph_mat, graph_feat = load_graph_info(f'../dataset/{opt.dataset}/graph_mat_5.cmu.pkl',
                                            f'../dataset/{opt.dataset}/user_tfidf_mat_9k.cmu.pkl')

    graph_feat = scipy2torch_sp(graph_feat).to(device)
    g_dgl = dgl.from_scipy(graph_mat).to(device)
    # g_dgl.ndata['text'] = graph_feat
    user_coor_df = load_user_coor(opt.dataset)
    user_coor_list = user_coor_df.values.tolist()
    user_loc_label = np.array(load_loc_label(opt.dataset))
    all_user = len(user_loc_label)
    print(user_loc_label)
    classLat_dict, classLon_dict = load_loc_coor(opt.dataset)

    train_slice = data_config[opt.dataset]['train']
    valid_slice = data_config[opt.dataset]['valid'] + train_slice
    test_slice = data_config[opt.dataset]['test'] + valid_slice

    # initial gcn model
    gcn = HighGCN(graph_feat.shape[1], opt.d_model, opt.class_num).to(device)
    cross_entropy = CrossEntropyLoss().to(device)
    optimizer = Adam(gcn.parameters(), lr=opt.lr)

    max_acc, max_mean, max_median = 0, 0, 0
    max_eps = 0
    for eps in range(opt.epoch):

        gcn.train()
        loss_num = 0
        iter = 0
        # Initialize parameters
        train_acc_num, test_acc_num = 0, 0
        train_user_num, test_user_num = 0, 0

        # Extract the corresponding data
        train_label = torch.tensor(user_loc_label[:train_slice])
        # print((tmp_train_label.shape))
        # tmp_train_idx = y_train[batch][:, 0]

        pred_result = gcn(g_dgl, graph_feat)
        # print(pred_coor)
        # print(tmp_train_label)

        # calculate loss
        loss = cross_entropy(pred_result[:train_slice], train_label.long().to(opt.device))
        loss_num += loss.item()
        iter += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate the acc
        pred_labels = extract_user_loc(pred_result[:train_slice])
        err_distances, acc, acc_161, mean_ed, median_ed = eval(pred_labels, user_loc_label[:train_slice],
                                                               user_coor_list[:train_slice],
                                                               classLat_dict, classLon_dict)
        print(f'epoch {eps}\n Accuracy: {acc}, ACC@161: {acc_161}, Mean Error: {mean_ed}, Median Error: {median_ed}, '
              f'Loss: {loss_num}')
        # train_acc = float(train_acc_num/len(X_train))

        gcn.eval()
        with torch.no_grad():

            pred_result = gcn(g_dgl, graph_feat)
            # print(user_coor_df[['lat', 'lon']].iloc[tmp_test_idx])
            # validate acc
            pred_labels = extract_user_loc(pred_result[valid_slice: test_slice])
            test_err_dist, test_acc, test_acc_161, mean_ed_test, median_ed_test = eval(pred_labels, user_loc_label[valid_slice: test_slice],
                                                                                       user_coor_list[valid_slice: test_slice],
                                                                                        classLat_dict, classLon_dict)
            print(f'Test Accuracy: {test_acc}, ACC@161: {test_acc_161}, Mean Error: {mean_ed_test}, Median Error: {median_ed_test}')

        # test_acc = float(test_acc_num/len(X_test))
        if test_acc_161 >= max_acc:
            max_acc = test_acc_161
            max_mean = mean_ed_test
            max_median = median_ed_test
            max_eps = eps
            best_err_dist = test_err_dist


        print(f'Max ACC@161: {max_acc}, Mean Error: {max_mean}, Median Error: {max_median}, eps: {max_eps}')
