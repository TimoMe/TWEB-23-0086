import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    ''' Dot-Product Attention '''
    def __init__(self, d_model, attn_dropout=0.3):
        # d_model是词语嵌入的长度
        super(DotProductAttention, self).__init__()
        self.q = nn.Linear(in_features=d_model, out_features=1, bias=False)
        self.L_k = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=True)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, k, v, mask=None):  # q/k/v.shape: (batchSize, seqLen, dim)
        # print('v shape:', v.shape)
        # transpose将K矩阵的seplen和dim转置
        # k = self.dropout1(torch.tanh(self.L(k)))
        k = torch.tanh(self.L_k(k))

        # print('q:', self.q)
        attn = self.q(k).squeeze(-1)  # attn.shape: (batchSize, q_seqLen, k_seqLen)
        # print('attn1:', attn)

        # 由于在attention之后需要对填充之后的多余的padding处理，这一部分就需要经过mask之后输出为负无穷，这样才能经过softmax之后mask的输出为0.
        if mask is not None:
            # mask==0,作为判断条件，将mask中值为1的部分对应的attn中的tensor（注意维度）替换为后面的数值-1e9
            # 使用-1e9代表负无穷
            attn = attn.masked_fill(mask == 1, -1e9)

        attn = F.softmax(attn, dim=-1).unsqueeze(1)
        # print('attention score shape:', attn.shape)
        output = torch.bmm(attn, v) # output.shape: (batchSize, q_seqLen, dim)
        # print('out', output.shape)
        return output, attn


def main():
    # TODO: your test code
    key = torch.randn(1, 4, 3)
    value = torch.randn(1, 4, 3)
    print('key:', key)
    print('value:', value)

    attn_layer = DotProductAttention(3)
    attn_output, attn = attn_layer(key, value)
    print(attn_output)
    print(attn)
    pass


if __name__ == '__main__':
    main()