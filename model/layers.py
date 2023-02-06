# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy, time


class ContextualEmbeddingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, gpu, max_seq_len):
        super(ContextualEmbeddingModel, self).__init__()

        self.char_feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.max_seq_len = max_seq_len

        self.convs = nn.ModuleList([\
                                    nn.Sequential(nn.Conv1d(self.char_feature_dim, self.hidden_dim, kernel_size=i, padding=0),
                                                  # nn.BatchNorm1d(),
                                                  nn.ReLU()
                                                  # nn.MaxPool1d(kernel_size=i)
                                                  )\
                                    for i in range(1, self.max_seq_len+1)\
                                   ])

        # self.cnn_layer_0 = nn.Conv1d(self.char_feature_dim, self.hidden_dim, kernel_size=1, padding=0)

        # self.cnn_layer_next = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i, padding=0) for i in range(3, 7)])
        # self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        # self.act_fun = nn.ReLU()
        # self.max_pool = nn.ModuleList([nn.MaxPool1d(kernel_size=i) for i in range(3, 7)])

        if self.gpu:

            self.convs = self.convs.cuda()

            # self.cnn_layer_0 = self.cnn_layer_0.cuda()
            # for i in range(6):
            #     self.cnn_layer_next[i] = self.cnn_layer_next[i].cuda()
            # # self.cnn_layer_next = self.cnn_layer_next.cuda()
            # self.batch_norm = self.batch_norm.cuda()
            # self.act_fun = self.act_fun.cuda()
            # for i in range(6):
            #     self.max_pool[i] = self.max_pool[i].cuda()
            # self.max_pool = self.max_pool.cuda()





    def forward(self, x):
        # x:[b, l, we]输入层必须和cnn的输入层保持维度一样
        assert x.size(-1) == self.char_feature_dim
        seq_len = x.size(1)
        # x:[b, we, l]
        x = x.transpose(1,2).contiguous()

        # x = self.cnn_layer_0(x)

        # 基础补齐组件
        # padding = torch.zeros(x.size(0), int(self.hidden_dim), 1)
        # if self.gpu:
        #     padding = padding.cuda()
        #
        # # 由range控制最终只用一次卷积完成对所有seq的操作
        # # 当kernel_size逐渐增大时，卷积得到的文本seq会越来越小，为了防止这个问题的出现，做补齐操作
        # # 本来range()里的参数是seq_len,这里换成4
        # for kernel_size in range(4):
        #     window_padding = torch.cat([padding for i in range(kernel_size + 2)], dim=2)
        #     # # [b, hidden_dim, l]--->[b, l, hidden_dim]
        #     # X_context = X_context.transpose(2,1).contiguous()
        #
        #     x = torch.cat([x, window_padding], dim=2)
        #
        #
        #     # 补齐后的X_context做卷积seq_len不会发生变化
        #     x = self.cnn_layer_next[kernel_size](x)
        #     x = self.batch_norm(x)
        #     x = self.act_fun(x)
        #     x = self.max_pool[kernel_size](x)
        #     # print(X_context.size())
        #     # input()
        # x = x.transpose(2, 1).contiguous()
        # 用不同kernel_size的卷积核对序列做卷积
        for i in range(seq_len):
            out = self.convs[i](x)
            out = nn.MaxPool1d(kernel_size=seq_len-i+1)(out)
            if i == 0:
                X_context = out

            # [b, we, l]
            else:
                X_context = torch.cat([X_context, out],dim=-1)

            # print("test!")
            # input()

        X_context = X_context.transpose(2, 1).contiguous()

        return X_context



# context embedding 和 gaz_embedding 分开
# 多层CNN提取multi-gram信息
class ContextModel0(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, gpu):
        super(ContextModel0,self).__init__()
        self.char_feature_dim = input_dim
        self.hidden_dim = hidden_dim
        # 第一层：把char_feature_dim转换到hidden_dim空间中来
        self.cnn_layer_0 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0)
        # 第二层，经过多个不同尺度的kernel_size作用后得到最终的contextual embedding
        self.cnn_contextual = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i, padding=0), nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)) for i in range(3, 7)])
        # nn.ReLU(inplace=True)换成nn.LeakyReLU
        self.padding = torch.zeros(batch_size, hidden_dim, 1)

        if gpu:
            self.cnn_layer_0 = self.cnn_layer_0.cuda()
            self.cnn_contextual = self.cnn_contextual.cuda()
            self.padding = self.padding.cuda()


    def forward(self, x):
        # 检查输入数据维度是否与初始化时对应
        assert x.size(-1) == self.char_feature_dim
        x_temp = x.transpose(2,1).contiguous()
        # 初层卷积
        X_context = self.cnn_layer_0(x_temp)
        # 多层卷积
        for kernel_size in range(4):
            # 拼接padding
            window_padding = torch.cat([self.padding for i in range(kernel_size+2)], dim=2)
            X_context = torch.cat([X_context, window_padding], dim=2)
            # 多层卷积
            X_context = self.cnn_contextual[kernel_size](X_context)

        X_context = X_context.transpose(2,1).contiguous()
        # print(X_context.size())
        # input()
        return X_context


# 得到已经包含上下文信息的context embedding，可以用这个表示first char 和 last char 以及 char pos
# 空洞卷积
class ContextModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, gpu):
        super(ContextModel1, self).__init__()
        self.char_feature_dim = input_dim
        self.hidden_dim = hidden_dim
        # 第一层：把char_feature_dim转换到hidden_dim空间中来
        self.cnn_layer_0 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.padding = torch.zeros(batch_size, hidden_dim, 1)

        # 空洞卷积
        self.dconv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=3//2, dilation=1),
            nn.SELU(), nn.AlphaDropout(p=0.05),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=3//2+1, dilation=2),
            nn.SELU(), nn.AlphaDropout(p=0.05),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=3//2+4, dilation=5),
            nn.SELU(), nn.AlphaDropout(p=0.05))

        # 第二层，经过多个不同尺度的kernel_size作用后得到最终的contextual embedding
        # self.cnn_contextual = nn.ModuleList([nn.Sequential(
        #     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i, padding=0), nn.BatchNorm1d(self.hidden_dim),
        #     nn.ReLU(inplace=True)) for i in range(3, 7)])
        # self.padding = torch.zeros(batch_size, hidden_dim, 1)

        if gpu:
            self.cnn_layer_0 = self.cnn_layer_0.cuda()
            # self.cnn_contextual = self.cnn_contextual.cuda()
            self.padding = self.padding.cuda()
            for i in self.dconv:
                i = i.cuda()
            self.dconv = self.dconv.cuda()

    def forward(self, x):
        # 检查输入数据维度是否与初始化时对应
        assert x.size(-1) == self.char_feature_dim
        x_temp = x.transpose(2, 1).contiguous()
        # seq_len = x_temp.size(-1)

        # 初层卷积
        X_context = self.cnn_layer_0(x_temp)

        # X_context = torch.cat([X_context, self.padding], dim=2)
        # print(X_context.size())
        # input()
        # 多层卷积
        # for kernel_size in range(4):
        #     # 拼接padding
        #     window_padding = torch.cat([self.padding for i in range(kernel_size + 2)], dim=2)
        #     X_context = torch.cat([X_context, window_padding], dim=2)
        #     # 多层卷积
        #     X_context = self.cnn_contextual[kernel_size](X_context)

        # X_context = X_context.transpose(0,1).transpose(1,2)
        # 空洞卷积
        X_context = self.dconv(X_context)
        # print(X_context.size())
        # input()

        X_context = X_context.transpose(2, 1).contiguous()
        # X_context = X_context.repeat(1,seq_len,1)
        # print(X_context.size())
        # input()
        return X_context

# 单层CNN提取特征（把embedding从feature_dim转换到hidden_dim空间中去）
class ContextModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, gpu):
        super(ContextModel2, self).__init__()
        self.char_feature_dim = input_dim
        self.hidden_dim = hidden_dim
        # # 第一层：把char_feature_dim转换到hidden_dim空间中来
        # self.cnn_layer_0 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0)
        # # 第二层，经过多个不同尺度的kernel_size作用后得到最终的contextual embedding
        # self.cnn_contextual = nn.ModuleList([nn.Sequential(
        #     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i, padding=0), nn.BatchNorm1d(self.hidden_dim),
        #     nn.ReLU(inplace=True)) for i in range(3, 7)])
        # # nn.ReLU(inplace=True)换成nn.LeakyReLU
        # self.padding = torch.zeros(batch_size, hidden_dim, 1)

        self.cnn_layer0 = nn.Conv1d(input_dim, self.hidden_dim, kernel_size=1, padding=0)

        if gpu:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            # self.cnn_contextual = self.cnn_contextual.cuda()
            # self.padding = self.padding.cuda()

    def forward(self, x):
        # 检查输入数据维度是否与初始化时对应
        assert x.size(-1) == self.char_feature_dim
        x_temp = x.transpose(2, 1).contiguous()
        # 初层卷积
        X_context = self.cnn_layer0(x_temp)
        # 多层卷积
        # for kernel_size in range(4):
        #     # 拼接padding
        #     window_padding = torch.cat([self.padding for i in range(kernel_size + 2)], dim=2)
        #     X_context = torch.cat([X_context, window_padding], dim=2)
        #     # 多层卷积
        #     X_context = self.cnn_contextual[kernel_size](X_context)
        X_context = torch.tanh(X_context)
        # [b_s, l, hid_dim]
        X_context = X_context.transpose(2, 1).contiguous()

        return X_context


# BiLSTM，同单层CNN
class ContextModel3(nn.Module):
    # biflag换成True会报错
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_layer=1, biflag=False):
        super(ContextModel3, self).__init__()

        # print(input_dim)
        # input()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_feature):
        hidden = None
        feature_out, hidden = self.lstm(input_feature, hidden)

        feature_out_d = self.drop(feature_out)
        return feature_out_d




class CNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout, gpu=True):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.gpu = gpu

        # 输入通道数和输出通道数
        self.cnn_layer0 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1) for i in range(self.num_layer-1)]
        self.drop = nn.Dropout(dropout)

        if self.gpu:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            for i in range(self.num_layer-1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()

    # input_feature:[batch_size, seq_len, embed_dim]
    def forward(self, input_feature):


        batch_size = input_feature.size(0)
        seq_len = input_feature.size(1)
        # [batch_size, embed_dim, seq_len]
        # contiguous()返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor
        input_feature = input_feature.transpose(2,1).contiguous()
        # [batch_size, output_channel/hidden_dim, seq_len-kernel_size+1]
        cnn_output = self.cnn_layer0(input_feature)  #(b,h,l)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer-1):
            # cnn_layers[i]: [batch_size, hidden_dim, seq_len-kernel_size+1]
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2,1).contiguous()
        return cnn_output








def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 对应Add和Norm
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    # size=d_model
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # Multi-headed attention forward
        # 首先经过norm，多头自注意力计算，然后经过Add和Norm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # FFN forward
        return self.sublayer[1](x, self.feed_forward)

# softmax(Q*k^T)/根号d_k
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # Q*K_T
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  ## (b,h,l,d) * (b,h,d,l)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   ##(b,h,l,l) * (b,h,l,d) = (b,h,l,d)


class MultiHeadedAttention(nn.Module):
    # head; hidden_dim
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 构建model_list（堆叠几层相同的）
        # 这里的4表示linear的层数也是head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query:[batch_size, seq_len, head, dim]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x:[b,h,l,d]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 多个头concat，再经过linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # d_model:hidden_dim
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    # x融入位置编码向量（该位置编码不可训练）
    def forward(self, x):
        # input: [batch_size, seq_Len, hidden_dim]
        x = x + autograd.Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)


class AttentionModel(nn.Module):
    "Core encoder is a stack of N layers"
    # d_input: input_dim
    # d_model: hidden_dim
    # d_ff: 2*hidden_dim
    # head: 4
    # num_layer: 4
    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        c = copy.deepcopy
        # attn0 = MultiHeadedAttention(head, d_input, d_model)
        # d_model=hidden_dim
        # d_ff=2*hidden_dim
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = PositionalEncoding(d_model, dropout)
        # layer0 = EncoderLayer(d_model, c(attn0), c(ff), dropout)
        layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.layers = clones(layer, num_layer)
        # layerlist = [copy.deepcopy(layer0),]
        # for _ in range(num_layer-1):
        #     layerlist.append(copy.deepcopy(layer))
        # self.layers = nn.ModuleList(layerlist)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input, d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # x: embedding (b,l,we)
        # 首先通过linear把input_dim转换到hidden_dim空间中
        # 然后融入位置编码
        x = self.posi(self.input2model(x))
        # 4个编码器层堆栈
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)




class char_word_emb(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True, biflag=True):
        super(char_word_emb, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        hidden = None
        word_out, hidden = self.lstm(input, hidden)
        word_out_d = self.drop(word_out)
        return word_out_d




class NERmodel(nn.Module):

    def __init__(self, model_type, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True, biflag=True):
        super(NERmodel, self).__init__()
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
            self.drop = nn.Dropout(dropout)

        if self.model_type == 'cnn':
            self.cnn = CNNmodel(input_dim, hidden_dim, num_layer, dropout, gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.attention_model = AttentionModel(d_input=input_dim, d_model=hidden_dim, d_ff=2*hidden_dim, head=4, num_layer=num_layer, dropout=dropout)
            for p in self.attention_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # if self.model_type == 'transformer-xl':


    # input: [b, l, embedding_dim]
    def forward(self, input, mask=None):

        if self.model_type == 'lstm':
            hidden = None
            # feature_out: [b, l, hidden_dim]
            feature_out, hidden = self.lstm(input, hidden)

            feature_out_d = self.drop(feature_out)

        if self.model_type == 'cnn':
            feature_out_d = self.cnn(input)

        if self.model_type == 'transformer':
            feature_out_d = self.attention_model(input, mask)

        return feature_out_d


# modify
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, hidden_dim, attn_dropout=0.1):
        super().__init__()
        # 根号d_k
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    # q: [n_head*batch_size, seq_len, d_k]
    # k: [n_head*batch_size, seq_len, d_k]
    # v: [n_head*batch_size, seq_len, d_v]
    def forward(self, q, k, v, mask=None):
        # torch.bmm()强制规定维度和大小相同，即前一个矩阵的列数=后一个矩阵的行数
        # Q*K^T
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # attn: [n_head*batch_size, seq_len, seq_len]
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # output: [n_head*batch_size, seq_len, d_v]
        output = torch.bmm(attn, v)

        # output是经过*V后的结果
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    # n_head=1; d_model = hidden_dim; d_k = d_v = hidden_dim//n_head
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # w*q, w*k, w*v
        # d_model = hidden_dim; hidden_dim
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), hidden_dim=self.d_v)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    # q = k = v = [b, l, hidden_dim]
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # 残差连接部分
        residual = q

        # 经过self.w_qs: [b, l, n_head*d_k]
        # 经过线性变换得到新的q,k,v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n_head*batch_size) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # 根据q,k,v进行放缩点积计算
        # output: [n_head*batch_size, seq_len, d_v]
        # attn: [n_head*batch_size, seq_len, seq_len]
        output, attn = self.attention(q, k, v, mask=mask)

        # 转换output的维度
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n_head*dv)

        # self.fc: [n_head*d_v, d_model]
        # output: [b, l, d_model]
        # residual: [b, l, hidden_dim]
        output = self.dropout(self.fc(output))
        # output: [b, l, hidden_dim](对应位置相加)
        output = self.layer_norm(output + residual)
        return output, attn

class GlobalGate(nn.Module):

    def __init__(self, hidden_dim, d_model):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.head = 1
        self.self_attention = MultiHeadAttention(self.head, self.d_model, self.hidden_dim//self.head, self.hidden_dim//self.head)

    # layer_output: [b, l, hidden_dim]
    # [b,l,4,e]
    def forward(self, layer_output, global_matrix=None):
        batch_size = layer_output.size(0)
        seq_len = layer_output.size(1)
        layer_output = layer_output.view(batch_size, seq_len, -1)
        new_global_matrix, _ = self.self_attention(layer_output, layer_output, layer_output)

        # [b, l, hidden_dim]
        return new_global_matrix
