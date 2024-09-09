# Social Media User Geolocation via Hybrid Attention
# https://github.com/zhengyuxiang/Social-Media-User-Geolocation-via-Hybrid-Attention

import torch.nn as nn
import torch
import numpy as np
import math
from Attention import DotProductAttention

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def generate_mask(word_num_list):
    attn_mask = torch.ones(len(word_num_list), max(word_num_list))
    for num in range(len(word_num_list)):
        word_num = word_num_list[num]
        assert word_num != 0
        attn_mask[num, :word_num] = torch.zeros(word_num)
    return attn_mask.to(bool)


class Word_Encoder(nn.Module):
    def __init__(self, d_model, out_dim, dropout=0.1):  # config是一个配置字典
        super(Word_Encoder, self).__init__()

        self.bi_lstm = nn.GRU(input_size=d_model,
                              hidden_size=out_dim,
                              num_layers=2,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=True)

        self.word_attn = DotProductAttention(2*out_dim, attn_dropout=dropout)
        self.fc0 = nn.Linear(2*out_dim, out_dim)

        self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, lengths, device):
        # print(src_seq.shape)
        # print(len(lengths))
        input_seq = pack_padded_sequence(src_seq, lengths, batch_first=True, enforce_sorted=False)
        out_emb, h_n = self.bi_lstm(input_seq)
        out_seq, lens_unpacked = pad_packed_sequence(out_emb, batch_first=True)
        word_attn_mask = generate_mask(lengths).to(device)
        sent_emb, attn_word = self.word_attn(out_seq, out_seq, word_attn_mask)
        out_seq = self.fc0(sent_emb)
        # out_seq = sent_emb

        return out_seq, attn_word


class Sentence_Encoder(nn.Module):
    def __init__(self, d_model, out_dim, dropout=0.1):  # config是一个配置字典
        super(Sentence_Encoder, self).__init__()
        self.bi_lstm = nn.GRU(input_size=d_model,
                              hidden_size=out_dim,
                              num_layers=2,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=True)
        self.sent_attn = DotProductAttention(2 * out_dim, attn_dropout=dropout)


    def forward(self, src_seq, sent_lengths, device):
        input_seq = pack_padded_sequence(src_seq, sent_lengths, batch_first=True, enforce_sorted=False)
        out_emb, h_n = self.bi_lstm(input_seq)
        out_seq, lens_unpacked = pad_packed_sequence(out_emb, batch_first=True)
        sent_attn_mask = generate_mask(sent_lengths).to(device)
        # print(out_seq[0])
        sent_emb, attn_sent = self.sent_attn(out_seq, out_seq, sent_attn_mask)
        # print(sent_emb[0], attn_sent[0])

        out_seq = sent_emb
        return out_seq, attn_sent


class TextNet(nn.Module):
    def __init__(self, word_emb_dim, hidden_dim, out_dim, dropout=0.4):
        super(TextNet, self).__init__()

        self.gru_word = Word_Encoder(word_emb_dim, hidden_dim, dropout=dropout)
        self.gru_user = Sentence_Encoder(hidden_dim, hidden_dim, dropout=dropout)

        # text mapping
        self.L1 = nn.Linear(in_features=2*hidden_dim, out_features=out_dim, bias=True)
        self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, text_src, word_lengths, sentence_lengths, device):
        word_emb = text_src  # b*l*d
        sent_emb, word_attn =  (self.gru_word(word_emb, word_lengths, device))  # b*l*d
        # print('sent emb shape:', sent_emb.shape)

        # reconstruct user tensor
        sent_emb_new = torch.zeros([len(sentence_lengths), max(sentence_lengths), sent_emb.size()[-1]]).to(device)
        start_len = 0
        for i in range(len(sentence_lengths)):
            sent_emb_new[i, :sentence_lengths[i], :] = sent_emb[start_len:start_len+sentence_lengths[i], 0, :].squeeze(1)
            start_len += sentence_lengths[i]

        final_emb, sent_attn = self.gru_user(sent_emb_new, sentence_lengths, device)  # b*d

        text_emb = self.dropout1(self.L1(final_emb.view([final_emb.size()[0], final_emb.size()[-1]])))
        # graph_emb = torch.stack(user_graph_emb)
        return text_emb


def text_test():
    word_emb = torch.rand([10, 5])
    word_embed_shape = word_emb.size()
    d_model = 5
    hidden_dim = 5
    out_dim = 2
    text_net = TextNet(word_emb, d_model, hidden_dim, out_dim, dropout=0.4).to('cpu')

    # text_src = [torch.randint(0, 10, [3, 2]), torch.randint(0, 10, [2, 3]), torch.randint(0, 10, [5, 4])]
    words_length = [3, 3, 2, 2, 2, 4, 4, 4, 4]
    text_src = torch.randint(0, 10, [9, 4])

    sentence_length = [2, 3, 4]
    text_emb = text_net(text_src, words_length, 'cpu', sentence_length)
    print(text_emb)


if __name__ == '__main__':
    text_test()