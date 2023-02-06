# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crf import CRF
from model.layers import NERmodel, GlobalGate, ContextualEmbeddingModel, ContextModel0, ContextModel2, ContextModel3, ContextModel1
# from transformers.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertModel

# 位置函数
# n_position:位置的最大值(instance的最大长度)
# d_hid:位置向量的维度
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    # position表示词在序列中的位置
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    # position表示某个固定位置
    # 求出该位置的位置编码的每一项
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # 求出每个位置的位置编码
    # [n_position, d_hid]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    # start:end:step
    # 对位置编码的偶数位置，使用sin
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class GazLSTM(nn.Module):
    def __init__(self, data):
        super(GazLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.posi_emb_dim = data.posi_emb_dim
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert

        self.HP_use_posi = data.HP_use_posi
        self.HP_use_context = data.HP_use_context
        self.HP_use_global_attention = data.HP_use_global_attention

        # 使用word的边界信息以及当前char的位置信息
        self.use_boundary = data.use_boundary
        self.use_char_pos = data.use_char_pos


        # 以下进行gaz,word,biword的向量表初始化
        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim)


        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))

        if self.use_biword:
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
            else:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), self.word_emb_dim)))

        # modify
        if self.use_char_pos:
            self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(20, self.posi_emb_dim), freeze=True)


        # 重新计算char_feature_dim
        char_feature_dim = self.word_emb_dim

        # print(char_feature_dim)

        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        # print(char_feature_dim)

        # 加上位置编码，则需要+posi_emb_dim
        if self.HP_use_posi:
            char_feature_dim += self.posi_emb_dim

        # print(char_feature_dim)

        """modify"""
        # self.cnn_layer_0 = nn.Conv1d(char_feature_dim, self.hidden_dim, kernel_size=1, padding=0)

        # 利用layers.py中的模块实现cnn提取全局信息

        # 多层CNN提取multi-gram
        self.cnn_context = ContextModel0(char_feature_dim, self.hidden_dim, data.HP_batch_size, self.gpu)

        # 空洞卷积
        # self.cnn_context = ContextModel1(char_feature_dim, self.hidden_dim, data.HP_batch_size, gpu=self.gpu)

        # 单层卷积
        # self.cnn_context = ContextModel2(char_feature_dim, self.hidden_dim, self.gpu)

        # LSTM
        # self.cnn_context = ContextModel3(char_feature_dim, self.hidden_dim)

        # 把位置编码加到cnn提取全局信息的过程中
        # self.cnn_contextual = ContextualEmbeddingModel(char_feature_dim, self.hidden_dim, self.gpu, data.posi_alphabet_size)

        """modify"""
        self.global_gate = GlobalGate(self.hidden_dim, self.hidden_dim)  # 4*self.gaz_emb_dim


        # 如果加上contextual_embedding，则需要+hidden_dim
        if self.HP_use_context:
            char_feature_dim += self.hidden_dim

        # print(char_feature_dim)

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        if self.use_count:
            char_feature_dim += 4*self.gaz_emb_dim

            # print(char_feature_dim)

            # modify
            if self.use_boundary:
                char_feature_dim += 8*self.word_emb_dim

            # print(char_feature_dim)

            if self.use_char_pos:
                char_feature_dim += 4*self.posi_emb_dim

            # print(char_feature_dim)
            # print(self.hidden_dim)
            # print("维度输出over")



        # print(char_feature_dim)

        # print(char_feature_dim-4*self.gaz_emb_dim-self.hidden_dim)
        # print(self.hidden_dim)
        # print('end')
        # input()

        """modify"""
        # 初始化的时候要初始化到最大长度
        # self.cnn_contextual = nn.ModuleList([ nn.Sequential(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i, padding=0), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(inplace=True)) for i in range(3, 7) ]) #原始[2,posi_alphabet_size+2]


        # 指定3种不同的模型作为序列建模层
        ## lstm model
        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            # self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim-self.word_emb_dim-self.biword_emb_dim, hidden_dim=lstm_hidden, num_layer=self.lstm_layer, biflag=self.bilstm_flag)
            self.NERmodel = NERmodel(model_type='lstm',
                                     input_dim=char_feature_dim,
                                     hidden_dim=lstm_hidden, num_layer=self.lstm_layer, biflag=self.bilstm_flag)

        ## cnn model
        if self.model_type == 'cnn':
            self.NERmodel = NERmodel(model_type='cnn', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout, gpu=self.gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout)


        # self-model
        if self.model_type == 'transformer-xl':
            self.NERmodel = NERmodel(model_type='transformer-xl', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout)


        '''modify'''
        if data.HP_use_posi:
            data.posi_alphabet_size += 1
            self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(data.posi_alphabet_size, self.posi_emb_dim), freeze=True)



        self.drop = nn.Dropout(p=data.HP_dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size+2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()

            if self.HP_use_posi:
                self.position_embedding = self.position_embedding.cuda()
            if self.use_char_pos:
                self.position_embedding = self.position_embedding.cuda()

            # self.cnn_layer_0 = self.cnn_layer_0.cuda()
            # self.cnn_contextual = self.cnn_contextual.cuda()
            # for i in range(4): # 本来是[posi_alphabet_size-1]
            #     self.cnn_contextual[i] = self.cnn_contextual[i].cuda()
            # modify
            self.cnn_context = self.cnn_context.cuda()

            # self.cnn_contextual = self.cnn_contextual.cuda()

            self.NERmodel = self.NERmodel.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()
            # modify
            self.global_gate = self.global_gate.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()




    def get_tags(self,gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        # layer_gaz:[batch_size, seq_len, 4, max_gazlist]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []
        # word_inputs:[batch_size, seq_len]一句话中每个char对应的id
        # word_embs:[batch_size, seq_len, embed_dim]
        word_embs = self.word_embedding(word_inputs)

        # 使用bigram增强语义向量表示
        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs,biword_embs],dim=-1)

        # 位置编码,modify
        if self.HP_use_posi:
            posi_inputs = torch.zeros(batch_size, seq_len).long()
            # 指示位置的index
            posi_inputs[:, :] = torch.LongTensor([ i+1 for i in range(seq_len)])
            if self.gpu:
                posi_inputs = posi_inputs.cuda()
            position_embs = self.position_embedding(posi_inputs)
            word_embs = torch.cat([word_embs, position_embs],dim=2)


        # transformer不需要drop
        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)   #(b,l,we)
        else:
            word_inputs_d = word_embs


        # print(word_inputs_d.size())
        # input()


        """CNN提取contextual_embedding"""
        # modify
        # if self.HP_use_context:
        #     word_inputs_d_temp = word_inputs_d.transpose(2, 1).contiguous()
        #     # word_embed_dim+biword_embed_dim--->hidden_dim
        #     X_context = self.cnn_layer_0(word_inputs_d_temp)
        #
        #     # print(self.hidden_dim)
        #     # print(X_context.size())
        #
        #     # 基础补齐组件
        #     padding = torch.zeros(batch_size, int(self.hidden_dim/2), 1)
        #     if self.gpu:
        #         padding = padding.cuda()
        #
        #     # 由range控制最终只用一次卷积完成对所有seq的操作
        #     # 当kernel_size逐渐增大时，卷积得到的文本seq会越来越小，为了防止这个问题的出现，做补齐操作
        #     # 本来range()里的参数是seq_len,这里换成4
        #     for kernel_size in range(4):
        #         window_padding = torch.cat([padding for i in range(kernel_size+2)], dim=2)
        #         # # [b, hidden_dim, l]--->[b, l, hidden_dim]
        #         # X_context = X_context.transpose(2,1).contiguous()
        #         X_context = torch.cat([X_context, window_padding], dim=2)
        #         # 补齐后的X_context做卷积seq_len不会发生变化
        #         X_context = self.cnn_contextual[kernel_size](X_context)
        #         # print(X_context.size())
        #         # input()
        #
        #     X_context = X_context.transpose(2,1).contiguous()


        if self.HP_use_context:
            # print(word_inputs_d.size())
            # input()
            X_context = self.cnn_context(word_inputs_d)
            # CNN后接global attention
            if self.HP_use_global_attention:
                X_context = self.global_gate(X_context)



        # 集成于layers.py文件中的网络模块
        # if self.HP_use_context:
        #     X_context = self.cnn_contextual(word_inputs_d)


        # False
        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1,1,1,1,1,self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  #(b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        # use gaz embedding（准备好gaz_embeds，和gaz对应）
        else:
            # layer_gaz: [batch_size, max_seq_len, 4, max_gazlist]，后面部分用0补齐
            # gaz_embeds: 加上embed_dim
            gaz_embeds = self.gaz_embedding(layer_gaz)


            # modify
            if self.use_boundary:
                word_bdaryl_embeds = self.word_embedding(gaz_first_idx)
                word_bdaryr_embeds = self.word_embedding(gaz_last_idx)
                # 简单拼接
                # gaz_multi_embeds = torch.cat([gaz_embeds, word_bdaryl_embeds, word_bdaryr_embeds], dim=-1)

            # modify
            if self.use_char_pos:
                # gaz_rel_pos = torch.LongTensor(gaz_rel_pos)
                # if self.gpu:
                #     gaz_rel_pos = gaz_rel_pos.cuda()

                # print(type(gaz_rel_pos))
                # print(gaz_rel_pos.device)
                # print(gaz_rel_pos.size())
                # input()

                # 因为是从1开始计数的
                char_pos_embeds = self.position_embedding(gaz_rel_pos)
                # gaz_multi_embeds = torch.cat([gaz_multi_embeds, char_pos_embeds], dim=-1)


            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
                # gaz_embeds_d = self.drop(gaz_multi_embeds)

                # modify
                if self.use_boundary:
                    word_bdaryl_embeds_d = self.drop(word_bdaryl_embeds)
                    word_bdaryr_embeds_d = self.drop(word_bdaryr_embeds)
                if self.use_char_pos:
                    char_pos_embeds_d = self.drop(char_pos_embeds)

            else:
                gaz_embeds_d = gaz_embeds
                # gaz_embeds_d = gaz_multi_embeds


                # modify
                if self.use_boundary:
                    word_bdaryl_embeds_d = word_bdaryl_embeds
                    word_bdaryr_embeds_d = word_bdaryr_embeds
                if self.use_char_pos:
                    char_pos_embeds_d = char_pos_embeds


            # 在最后增加一个维度，在各个维度上重复指定次数
            # gaz_mask_input: [batch_size, max_seq_len, 4, max_gaz_num],实际部分全是0，其他全是1
            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.gaz_emb_dim)
            # 把embedding中没有意义的部分(mask中对应1)全部用0替代
            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  #(b,l,4,g,ge)  ge:gaz_embed_dim


            # modify
            if self.use_boundary:
                gaz_char_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.word_emb_dim)
                word_bdaryl_embeds = word_bdaryl_embeds_d.data.masked_fill_(gaz_char_mask.data, 0)
                word_bdaryr_embeds = word_bdaryr_embeds_d.data.masked_fill_(gaz_char_mask.data, 0)
            if self.use_char_pos:
                gaz_pos_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.posi_emb_dim)
                char_pos_embeds = char_pos_embeds_d.data.masked_fill_(gaz_pos_mask.data, 0)


        # 在gaz_embeds前面加上权重
        # 对first_char_embedding和last_char_embedding做同样操作
        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  #(b, l, 4, gn)--->(b, l, 4, 1)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  #(b,l,1,1)
            # gaz_count: [b, l, 4, max_gazlist]
            # 权重的计算：每个word在一个instance中出现概率
            weights = gaz_count.div(count_sum)  #(b,l,4,g)
            weights = weights*4
            # [b,l,4,g,1]
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights*gaz_embeds  #(b,l,4,g,e)

            # modify
            word_bdaryl_embeds = weights*word_bdaryl_embeds
            word_bdaryr_embeds = weights*word_bdaryr_embeds

            # modify
            # gaz_embed_dim---> + 2*word_embed_dim + pos_embed_dim
            if self.use_boundary:
                gaz_embeds = torch.cat([gaz_embeds, word_bdaryl_embeds, word_bdaryr_embeds], dim=-1)
            if self.use_char_pos:
                gaz_embeds = torch.cat([gaz_embeds, char_pos_embeds], dim=-1)


            gaz_embeds = torch.sum(gaz_embeds, dim=3)  #(b,l,4,e)
            # .div(max_gaz_num)
            # # forward
            # gaz_embeds = self.global_gate(gaz_embeds)

        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  #(b,l,4,ge)/(b,l,4,1)

        gaz_embeds_cat = gaz_embeds.view(batch_size,seq_len,-1)  #(b,l,4*ge)

        # print(word_inputs_d.size())
        # print(gaz_embeds_cat.size())
        # print(X_context.size())
        # input()



        # 把经过bigram和dropout操作的word_input拿过来和gaz_embed融合
        word_input_cat = torch.cat([word_inputs_d, gaz_embeds_cat], dim=-1)  #(b,l,we+4*ge)

        # 第一种连接方式：word_embedding+gaz_embedding+context_embedding
        if self.HP_use_context:
            word_input_cat = torch.cat([word_input_cat, X_context], dim=-1)


        # 第二种连接方式：context_embedding+gaz_embedding(因为context_embedding相当于是融合了全局上下文信息的word_embedding)
        # if self.HP_use_context:
        #     word_input_cat = torch.cat([X_context, gaz_embeds_cat], dim=-1)


        # cat bert feature(False)
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:,1:-1,:]
            word_input_cat = torch.cat([word_input_cat, outputs], dim=-1)


        # print(word_input_cat.size())
        # input()
        # 序列建模层
        feature_out_d = self.NERmodel(word_input_cat)


        # modify
        # 经过BiLSTM计算后，通过global attention的强化
        # feature_out_d = self.global_gate(feature_out_d)


        # tags: [b,l,tag_size]
        tags = self.hidden2tag(feature_out_d)

        return tags, gaz_match


    # train
    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos):
        # tags:[b, l, tag_size]
        tags, _ = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos)
        # mask:[b, l]，1指示每个instance实际长度，0是用于补齐
        # batch_label:[b, l]，保存的是instance的label_Ids，初始化的时候全为0
        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq


    # dev
    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,layer_gaz, gaz_count,gaz_chars, gaz_mask,gazchar_mask, mask, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos):

        tags, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq, gaz_match







