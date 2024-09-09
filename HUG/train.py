import argparse
import numpy as np

import dgl
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe

from model import HUG
from utils import load_text_info, load_graph_info, load_user_coor, load_loc_coor, load_loc_label
from eval import eval
from tqdm import tqdm
from haversine import haversine


data_config = {
    'cmu':{
        'train': 5685,
        'valid': 1895,
        'test': 1895
    },
    'twitter-us':{
        'train': 429200,
        'valid': 10000,
        'test': 10000
    }
}
GLOVE_DIM = 100

GLOVE = GloVe(name='twitter.27B', dim=GLOVE_DIM, cache='../dataset/glove')


def text_align(nodes, user_text_dict):
    words_length = []
    sentence_length = []
    text_src = []
    for node in nodes:
        node = int(node)
        words_length += [len(text) for text in user_text_dict[node]]
        text_src += user_text_dict[node]
        sentence_length.append(len(user_text_dict[node]))

    text_mat = []
    for i in range(len(words_length)):
        if len(text_src[i]) == 0:
            print('error!!')
            continue
        text_tensor = GLOVE.get_vecs_by_tokens(text_src[i])
        text_mat += [text_tensor]
    text_mat = pad_sequence(text_mat, batch_first=True)

    return text_mat, words_length, sentence_length


def extract_user_loc(prob_mat):
    prob_mat1 = prob_mat.cpu()
    pred_loc = torch.max(prob_mat1, 1)[1]
    pred_loc_list = pred_loc.numpy().tolist()
    return pred_loc_list


def scipy2torch_sp(sp_mat):
    sp_mat_coo = sp_mat.tocoo()
    row = sp_mat_coo.row
    col = sp_mat_coo.col
    data = sp_mat_coo.data

    torch_idx = torch.tensor(np.array([row, col]))
    sp_mat_torch = torch.sparse_coo_tensor(indices=torch_idx, values=data, size=sp_mat.shape)

    return sp_mat_torch


def location_distance_matrix(class_lat_median, class_lon_median):
    distance_mtx = torch.FloatTensor(len(class_lon_median), len(class_lon_median)).cuda(0)
    for i, weight_i in enumerate(class_lon_median):
        for j, weight_j in enumerate(class_lon_median):
            if i == j:
                distance_mtx[i][j] = float('inf')
            else:
                distance_mtx[i][j] = haversine(
                    (float(class_lat_median[i]), float(class_lon_median[i])),
                    (float(class_lat_median[j]), float(class_lon_median[j])))
    return distance_mtx


def location_regularization(weight_mtx, distance_mtx):
    t1 = weight_mtx.unsqueeze(1).expand(len(weight_mtx), len(weight_mtx), weight_mtx.shape[1])
    t2 = weight_mtx.unsqueeze(0).expand(len(weight_mtx), len(weight_mtx), weight_mtx.shape[1])
    d = (t1 - t2).pow(2).sum(2) / distance_mtx
    return torch.sum(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, choices=['cmu', 'twitter-us', 'twitter-world'], default='cmu')  # choose dataset

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-accumulate_steps', type=int, default=35)
    parser.add_argument('-b', '--batch_size', type=int, default=512)

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-hidden_dim_t', type=int, default=100)
    parser.add_argument('-hidden_dim_g', type=int, default=128)

    parser.add_argument('-head_num', type=int, default=8)
    # parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-dropout', type=float, default=0.4)

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

    graph_feat = scipy2torch_sp(graph_feat)
    g_dgl = dgl.from_scipy(graph_mat)
    g_dgl.ndata['text'] = graph_feat
    user_coor_df = load_user_coor(opt.dataset)
    user_loc_label = np.array(load_loc_label(opt.dataset))
    all_user = len(user_loc_label)
    print(user_loc_label)
    classLat_dict, classLon_dict = load_loc_coor(opt.dataset)
    word_emb_dim = GLOVE_DIM
    # initial HUG model
    model = HUG(opt, word_emb_dim, graph_feat.shape[1], class_num=len(classLon_dict)).to(device)
    cross_entropy = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    # for para in model.fc.parameters():
    #     print((para))
    distance_mat = location_distance_matrix(classLat_dict, classLon_dict)

    # dataloader
    graph_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_ids = [i for i in range(data_config[opt.dataset]['train'])]
    dataloader = dgl.dataloading.DistNodeDataLoader(g_dgl, train_ids, graph_sampler,
                                                    batch_size=opt.batch_size, shuffle=True)
    valid_ids = [i for i in range(len(train_ids), len(train_ids)+data_config[opt.dataset]['valid'])]
    dataloader_val = dgl.dataloading.DistNodeDataLoader(g_dgl, valid_ids, graph_sampler,
                                                        batch_size=opt.batch_size, shuffle=False)
    test_ids = [i for i in range(len(train_ids)+len(valid_ids), all_user)]
    dataloader_test = dgl.dataloading.DistNodeDataLoader(g_dgl, test_ids, graph_sampler,
                                                         batch_size=opt.batch_size, shuffle=False)
    max_val_acc = 0
    best_val_info = {}
    best_test_info = {}
    for epoch in range(opt.epoch):
        pred_labels = []
        true_labels = []
        user_true_coor_list = []
        loss_num = 0
        iter = 0
        model.train()
        for step, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader), desc='Training'):
            # batch of graph
            blocks = [b.to(device) for b in blocks]
            node_feat = blocks[0].srcdata['text']

            # batch of text
            text_seq, words_len, sentence_len = text_align(output_nodes, word_emb_train)

            # forward
            pred_mat = model(text_seq.to(device), words_len, sentence_len, blocks, node_feat, device=opt.device)

            pred_labels_tmp = extract_user_loc(pred_mat)
            labels = user_loc_label[output_nodes.cpu().tolist()]

            pred_labels += pred_labels_tmp
            true_labels += labels.tolist()
            user_true_coor_list += user_coor_df[['lat', 'lon']].iloc[output_nodes.tolist()].values.tolist()

            # loss
            for param in model.fc.parameters():
                fc_weight = param
                break
            loss_R = 0.005*location_regularization(fc_weight, distance_mat)
            loss = cross_entropy(pred_mat, torch.tensor(labels).long().to(device)) #+ loss_R
            loss_num += loss.item()
            iter += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err_distances, acc, acc_161, mean_ed, median_ed = eval(pred_labels, true_labels, user_true_coor_list, classLat_dict, classLon_dict)
        print(f'epoch {epoch}\n Accuracy: {acc}, ACC@161: {acc_161}, Mean Error: {mean_ed}, Median Error: {median_ed}, '
              f'Loss: {loss_num/iter}')
    #
        model.eval()
        pred_labels_val, pred_labels_test = [], []
        true_labels_val, true_labels_test = [], []
        user_true_coor_val, user_true_coor_test = [], []
        loss_num_val = 0
        iter_val = 0
        best_err_dists = []
        with torch.no_grad():
            # valid evaluation
            for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader_val):
                # batch of graph
                blocks = [b.to(device) for b in blocks]
                node_feat = blocks[0].srcdata['text']

                # batch of text
                text_seq, words_len, sentence_len = text_align(output_nodes, word_emb_valid)

                # forward
                pred_mat = model(text_seq.to(device), words_len, sentence_len, blocks, node_feat, device=opt.device)
                pred_labels_tmp = extract_user_loc(pred_mat)

                labels = user_loc_label[output_nodes.cpu().tolist()]
                pred_labels_val += pred_labels_tmp
                true_labels_val += labels.tolist()
                user_true_coor_val += user_coor_df[['lat', 'lon']].iloc[output_nodes.tolist()].values.tolist()

                # loss
                loss = cross_entropy(pred_mat, torch.tensor(labels).long().to(device))
                loss_num_val += loss.item()
                iter_val += 1

            # test evaluation
            for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader_test):
                # batch of graph
                blocks = [b.to(device) for b in blocks]
                node_feat = blocks[0].srcdata['text']
                # batch of text
                text_seq, words_len, sentence_len = text_align(output_nodes, word_emb_test)
                # forward
                pred_mat = model(text_seq.to(device), words_len, sentence_len, blocks, node_feat, device=opt.device)
                pred_labels_tmp = extract_user_loc(pred_mat)

                labels = user_loc_label[output_nodes.cpu().tolist()]

                pred_labels_test += pred_labels_tmp
                true_labels_test += labels.tolist()
                user_true_coor_test += user_coor_df[['lat', 'lon']].iloc[output_nodes.tolist()].values.tolist()

            err_dists_val, acc_val, acc_161_val, mean_ed_val, median_ed_val = eval(pred_labels_val, true_labels_val,
                                                                                   user_true_coor_val,
                                                                                   classLat_dict, classLon_dict)
            err_dists_test, acc_test, acc_161_test, mean_ed_test, median_ed_test = eval(pred_labels_test, true_labels_test,
                                                                                        user_true_coor_test,
                                                                                        classLat_dict, classLon_dict)

        print(f'Epoch: {epoch}, Val Acc: {acc_val}, Val Acc@161: {acc_161_val}, Val Mean Error: {mean_ed_val}, '
              f'Val Median Error: {median_ed_val}, Val Loss: {loss_num_val / iter_val}')

        # test evaluation
        print(f'Test Acc: {acc_test}, Test Acc@161: {acc_161_test}, Test Mean Error: {mean_ed_test}, '
              f'Test Median Error: {median_ed_test}')

        if acc_161_val > max_val_acc:
            max_val_acc = acc_161_val
            best_val_info = {'epoch': epoch, 'acc': acc_val, 'acc_161': acc_161_val, 'mean_ed': mean_ed_val,
                             'median_ed': median_ed_val}
            best_test_info = {'acc': acc_test, 'acc_161': acc_161_test, 'mean_ed': mean_ed_test,
                              'median_ed': median_ed_test}
            best_err_dists = err_dists_test
        print('best test result', best_test_info)
    print('opt information:', opt)
    print('best geolocation result')
    print('Valid', best_val_info)
    print('Test', best_test_info)
    # save error distances as csv file
    import csv
    with open(f'./result/{opt.dataset}_err_dists_HUG_headnum_{opt.head_num}.csv', 'w') as f:
        wt = csv.writer(f)
        wt.writerow(best_err_dists)

