# three-layer MLP
# concat tfidf feature of text and adjacency matrix of graph

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.utils.data.dataset as Dataset

import numpy as np
import pickle as pkl
import csv

from eval import eval as evaluate
from utils import load_user_coor, load_loc_coor, load_loc_label


class ThreeLayerMLP(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=feat_dim,
                             out_features=hidden_dim,
                             bias=True)
        self.fc3 = nn.Linear(in_features=hidden_dim,
                             out_features=output_dim,
                             bias=True)
        self.relu = nn.ReLU()
        self.three_fc = nn.Sequential(self.fc1, self.relu, self.fc3)

    def forward(self, x):
        x = self.three_fc(x)
        return x



def get_features(text_mat, adj_mat):
    # text_mat: (batch_size, num_words)
    # adj_mat: (batch_size, num_nodes)
    # return: (batch_size, num_words + num_nodes)
    features = torch.cat((torch.tensor(text_mat.todense()), torch.tensor(adj_mat.todense())), dim=-1) # (batch_size, num_nodes, num_words + num_nodes)

    return features


class GeoDataset(Dataset.Dataset):
    def __init__(self, nodes, text_mat, adj_mat, target):
        self.nodes = nodes
        self.text_mat = text_mat
        self.adj_mat = adj_mat
        self.target = target

    def __getitem__(self, index):
        return self.nodes[index], get_features(self.text_mat[index], self.adj_mat[index]), self.target[index]
        # return self.nodes[index], scipy2torch_sp(self.adj_mat[index]), self.target[index]

    def __len__(self):
        return len(self.nodes)


def extract_user_loc(prob_mat):
    prob_mat1 = prob_mat.cpu()
    pred_loc = torch.max(prob_mat1, 1)[1]
    pred_loc_list = pred_loc.numpy().tolist()
    return pred_loc_list


def train(train_loader, model, criterion, optimizer, device, user_coor_df, class_lat, class_lon):
    model.train()
    loss_num = 0
    pred_loc = []
    true_loc = []
    user_true_coor_list = []
    for batch_idx, (nodes, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        loss = criterion(output, torch.tensor(target).long())
        pred_loc += extract_user_loc(output)
        true_loc += target.cpu().numpy().tolist()

        user_true_coor_list += user_coor_df[['lat', 'lon']].iloc[nodes.tolist()].values.tolist()

        loss_num += loss.item()
        loss.backward()
        optimizer.step()

    err_dists, acc, acc_161, mean_ed, median_ed = evaluate(pred_loc, true_loc, user_true_coor_list, class_lat, class_lon)
    print('Train avg loss: {:.4f}, acc: {:.4f}, acc_161: {:.4f}, mean_ed: {:.4f}, median_ed: {:.4f}'
          .format(loss_num / len(train_loader), acc, acc_161, mean_ed, median_ed))
    return err_dists, acc, acc_161, mean_ed, median_ed


def test(test_loader, model, criterion, device, user_coor_df, class_lat, class_lon):
    model.eval()
    test_loss = 0
    correct = 0
    pred_loc = []
    true_loc = []
    user_true_coor_list = []
    with torch.no_grad():
        for nodes, data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
            test_loss += criterion(output, target).item() # sum up batch loss

            pred_loc += extract_user_loc(output)
            true_loc += target.cpu().numpy().tolist()
            user_true_coor_list += user_coor_df[['lat', 'lon']].iloc[nodes.tolist()].values.tolist()

        # evaluate
        err_dists, acc, acc_161, mean_ed, median_ed = evaluate(pred_loc, true_loc, user_true_coor_list,
                                                               class_lat, class_lon)

        print('Test avg loss: {:.4f}, acc: {:.4f}, acc_161: {:.4f}, mean_ed: {:.4f}, median_ed: {:.4f}'
              .format(test_loss / len(test_loader), acc, acc_161, mean_ed, median_ed))
        return err_dists, acc, acc_161, mean_ed, median_ed


def scipy2torch_sp(sp_mat):
    sp_mat_coo = sp_mat.tocoo()
    row = sp_mat_coo.row
    col = sp_mat_coo.col
    data = sp_mat_coo.data

    torch_idx = torch.tensor(np.array([row, col]))
    sp_mat_torch = torch.sparse_coo_tensor(indices=torch_idx, values=data, size=sp_mat.shape)

    return sp_mat_torch


if __name__ == '__main__':
    epoch = 20
    # load graph & text feature
    graph_feat = pkl.load(open('../dataset/cmu/graph_mat_5.cmu.pkl', 'rb'))
    text_feat = pkl.load(open('../dataset/cmu/user_tfidf_mat_9k.cmu.pkl', 'rb'))

    # transfer graph_feat & text_feat to torch.sparse_coo_tensor
    # graph_feat = scipy2torch_sp(graph_feat)
    # text_feat = scipy2torch_sp(text_feat)
    # feat_mat = get_features(text_feat, graph_feat)

    # load label & coordinate
    user_loc_list = load_loc_label('cmu')
    user_coor_df = load_user_coor('cmu')
    class_lat, class_lon = load_loc_coor('cmu')

    # train_loader & test_loader
    train_nodes = [i for i in range(5685)]
    train_label = user_loc_list[:5685]
    train_dataset = GeoDataset(train_nodes, text_feat[train_nodes], graph_feat[train_nodes], train_label)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    valid_nodes = [i for i in range(5685, 7580)]
    valid_label = user_loc_list[5685: 7580]
    valid_dataset = GeoDataset(valid_nodes, text_feat[valid_nodes], graph_feat[valid_nodes], valid_label)
    valid_loader = DataLoader(valid_dataset, batch_size=200, shuffle=True)

    test_nodes = [i for i in range(7580, 9475)]
    test_label = user_loc_list[-1895:]
    test_dataset = GeoDataset(test_nodes, text_feat[test_nodes], graph_feat[test_nodes], test_label)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)

    # model initialize
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    model = ThreeLayerMLP(feat_dim=text_feat.shape[1] + graph_feat.shape[1],
                          hidden_dim=300,
                          output_dim=len(class_lat))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    max_valid = 0
    test_state = []
    err_dists_test = []

    for i in range(epoch):
        print('epoch: {:d}'.format(i))
        # train
        results = train(train_loader, model, criterion, optimizer, device, user_coor_df, class_lat, class_lon)
        err_dists, acc, acc_161, mean_ed, median_ed = results

        # valid
        valid_results = test(valid_loader, model, criterion, device, user_coor_df, class_lat, class_lon)
        valid_err_dists, valid_acc, valid_acc_161, valid_mean_ed, valid_median_ed = valid_results
        # test
        test_results = test(test_loader, model, criterion, device, user_coor_df, class_lat, class_lon)
        test_err_dists, test_acc, test_acc_161, test_mean_ed, test_median_ed = test_results
        if valid_acc_161 > max_valid:
            max_valid = valid_acc_161
            test_state = [i, test_acc, test_acc_161, test_mean_ed, test_median_ed]
            err_dists_test = test_err_dists
            # torch.save(model.state_dict(), './result/mlp_model_params.pkl')

    # output best result
    print('best result: epoch:{:d}, acc: {:.4f}, acc_161: {:.4f}, mean_ed: {:.4f}, median_ed: {:.4f}'
          .format(test_state[0], test_state[1], test_state[2], test_state[3], test_state[4]))

    # save error distances as csv file
    with open('./result/mlp_test_err_dists.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(err_dists_test)
