# batch method implementation of GAT

import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch


class HomoGAT(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, num_head=4):
        super(HomoGAT, self).__init__()
        self.fc = nn.Linear(in_features=feat_dim,
                            out_features=hidden_dim,
                            bias=False)
        activation = nn.ELU()
        self.dropout = nn.Dropout(p=0.4)

        self.gatconv1 = GATConv(in_feats=hidden_dim,
                                out_feats=int(hidden_dim / num_head),
                                num_heads=num_head,
                                activation=activation,
                                allow_zero_in_degree=True)
        # self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.gatconv2 = GATConv(in_feats=hidden_dim,
                                out_feats=output_dim,
                                num_heads=num_head,
                                activation=activation,
                                allow_zero_in_degree=True)

    # onehot 为用户文本的igr筛选出的文本onehot
    def forward(self, blocks, x_feat):
        x_feat = self.dropout(torch.tanh(self.fc(x_feat)))

        gat_emb0 = self.gatconv1(blocks[0], x_feat)
        gat_emb0 = gat_emb0.view([gat_emb0.size()[0], gat_emb0.size()[1] * gat_emb0.size()[-1]])

        # gat_emb0 = self.batch_norm(gat_emb0)

        gat_emb = self.dropout(self.gatconv2(blocks[1], gat_emb0))
        graph_emb = torch.mean(gat_emb, dim=-2)

        return graph_emb




