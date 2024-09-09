import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F

from Text_model import TextNet
from GAT import HomoGAT
from Attention import DotProductAttention


class HUG(nn.Module):
    def __init__(self, args, word_emb_dim, graph_feat_dim, class_num):
        super(HUG, self).__init__()

        self.text_net = TextNet(word_emb_dim, args.hidden_dim_t, args.d_model, args.dropout)
        self.gat = HomoGAT(graph_feat_dim, args.hidden_dim_g, args.d_model, args.head_num)
        self.fusion_attn = DotProductAttention(args.d_model, attn_dropout=args.dropout)
        self.fc = nn.Linear(in_features=args.d_model, out_features=class_num, bias=True)
        self.batch_norm = torch.nn.BatchNorm1d(args.d_model)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text_seq, word_length, sentence_num_list, blocks, graph_feature, device):
        text_rep = self.text_net(text_seq, word_length, sentence_num_list, device)
        graph_rep = self.gat(blocks, graph_feature)
        # print(text_rep.shape)
        # print(graph_rep.shape)
        # print('graph shape:', graph_rep.shape)
        fusion_rep = torch.cat([text_rep.unsqueeze(1), graph_rep.unsqueeze(1)], dim=1)
        # print('fusion dim:', fusion_rep.shape)
        fusion_rep, fusion_attn = self.fusion_attn(fusion_rep, fusion_rep)
        fusion_rep = self.dropout(self.batch_norm(fusion_rep.squeeze(1)))
        logits = self.fc(fusion_rep)
        return logits

