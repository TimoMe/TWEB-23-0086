import dgl
import os
from dgl.nn import SGConv
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import gensim
import pickle as pkl
import numpy as np
from utils import load_loc_coor, load_user_coor, load_loc_label

sys.path.append("..")
from eval import eval



class SGC(nn.Module):
    def __init__(self, in_feats, hidden_size, layer_num, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(in_feats, hidden_size, k=layer_num)
        self.fc = nn.Linear(hidden_size, num_classes, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(in_feats)

    def forward(self, g, features):
        features = self.layer_norm(features)
        x = self.conv1(g, features)
        x = self.dropout(torch.relu(x))
        x = self.fc(x)
        return x
    

def extract_user_loc(prob_mat):
    prob_mat1 = prob_mat.cpu()
    pred_loc = torch.max(prob_mat1, 1)[1]
    pred_loc_list = pred_loc.numpy().tolist()
    return pred_loc_list


def train(model, g, features, labels, train_num, user_coor_list, class_lat, class_lon, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = loss_func(logits[:train_num], labels[:train_num])
    loss.backward()
    optimizer.step()

    print('loss: {}'.format(loss.item()))

    # evaluate
    pred_loc = extract_user_loc(logits[:train_num])
    err_dists, acc, acc_161, mean_ed, median_ed = eval(pred_loc, labels[:train_num].tolist(), user_coor_list, class_lat, class_lon)
    return err_dists, acc, acc_161, mean_ed, median_ed
    

def test(model, g, features, labels, eval_slice, user_coor_list, class_lat, class_lon,):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        pred_loc = extract_user_loc(logits[eval_slice])
        # evaluate
        err_dists, acc, acc_161, mean_ed, median_ed = eval(pred_loc, labels, user_coor_list, class_lat, class_lon)
    return err_dists, acc, acc_161, mean_ed, median_ed


def load_doc2vec_feature(user_num, doc2vec_model_file):
    """
        doc2vec_model_file: the file that including all doc2vec features of the raw content.
    """
    # load model
    model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model_file)

    # train data features
    feature_list = list()
    for i in range(user_num):
        feature_list.append(model.docvecs[i])
    X_feature = np.array(feature_list)

    return X_feature


def load_data(root_path, dataset):
    g = pkl.load(open(os.path.join(root_path, 'graph_mat_5.cmu.pkl'), 'rb'))
    # transfer G TO dgl.graph
    g_dgl = dgl.from_scipy(g)
    g_dgl = dgl.add_self_loop(g_dgl)
    
    loc_labels = torch.tensor(load_loc_label(dataset)).long()
    node_feature = load_doc2vec_feature(user_num=len(loc_labels), doc2vec_model_file=f'./{dataset}/model_dim_512_epoch_40_cmu.bin')
    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    user_coor_df, train_num, valid_num, test_num = load_user_coor(dataset)
    user_coor_list = user_coor_df.values.tolist()
    class_lat, class_lon = load_loc_coor(dataset)

    return g_dgl, node_feature, loc_labels, train_num, valid_num, user_coor_list, class_lat, class_lon
    


if __name__ == '__main__':
    # load data
    root_path = '../../dataset/cmu'
    g, features, labels, train_num, valid_num, user_coor_list, class_lat, class_lon = load_data(root_path, 'cmu')
    print('train number: {}, valid number: {}'.format(train_num, valid_num))
    valid_slice = [i for i in range(train_num, train_num+valid_num)]
    test_slice = [i for i in range(train_num+valid_num, len(labels))]
    # model initialize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    class_num = 256 # twitter-us
    model = SGC(features.shape[1], 512, 2, 256).to(device)
    # train and test
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    best_valid_acc_161 = 0
    best_test_result = []
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500],
                                                gamma=0.1, last_epoch=-1)
    
    for epoch in range(1000):
        
        print('Epoch: {}'.format(epoch))
        train_result = train(model, g.to(device), features.to(device), labels.to(device), train_num, 
                             user_coor_list[:train_num], class_lat, class_lon, optimizer, loss_func)
        train_err_dists, train_acc, train_acc_161, train_mean_ed, train_median_ed = train_result
        print('Train:  acc {:.4f}, acc_161 {:.4f}, mean_ed {:.4f}, median_ed {:.4f}'
              .format( train_acc, train_acc_161, train_mean_ed, train_median_ed))
        scheduler.step()
        valid_data = test(model, g.to(device), features.to(device), labels[train_num: train_num+valid_num], valid_slice,
                          user_coor_list[train_num: train_num+valid_num], 
                          class_lat, class_lon)
        valid_err_dists, valid_acc, valid_acc_161, valid_mean_ed, valid_median_ed = valid_data
        print('Valid:  acc {:.4f}, acc_161 {:.4f}, mean_ed {:.4f}, median_ed {:.4f}'
              .format( valid_acc, valid_acc_161, valid_mean_ed, valid_median_ed))

        test_data = test(model, g.to(device), features.to(device), labels[train_num+valid_num:], test_slice,
                         user_coor_list[train_num+valid_num:], 
                         class_lat, class_lon)
        test_err_dists, test_acc, test_acc_161, test_mean_ed, test_median_ed = test_data
        print('Test: acc {:.4f}, acc_161 {:.4f}, mean_ed {:.4f}, median_ed {:.4f}'
              .format( test_acc, test_acc_161, test_mean_ed, test_median_ed))
        
        if valid_acc_161 > best_valid_acc_161:
            best_valid_acc_161 = valid_acc_161
            best_test_result = [test_err_dists, test_acc, test_acc_161, test_mean_ed, test_median_ed]
            print('Best test: acc {:.4f}, acc_161 {:.4f}, mean_ed {:.4f}, median_ed {:.4f}'
                  .format(best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4]))
    print('Best test: acc {:.4f}, acc_161 {:.4f}, mean_ed {:.4f}, median_ed {:.4f}'
          .format(best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4]))
    best_err_dist = best_test_result[0]
    import csv
    with open('./cmu/SGC_results.txt', 'w') as f:
        wt = csv.writer(f)
        wt.writerow(best_err_dist)
