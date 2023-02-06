# -*- coding: utf-8 -*-

import sys
import numpy as np
import re
from utils.alphabet import Alphabet
from transformers.models.bert.tokenization_bert import BertTokenizer
NULLKEY = "-null-"

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(num_layer, input_file, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        # 得到一行的内容
        line = in_lines[idx]
        # 得到words,biwords,labels,chars及其Ids
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            # 构建biword
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_index = biword_alphabet.get_index(biword)
            biword_Ids.append(biword_index)
            label_Ids.append(label_alphabet.get_index(label))

            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)

            # char_padding_size=-1
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass

            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        # 处理一句话,得到instence及Ids
        else:
            # max_sent_length=250
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):

                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []

                # 一个instance的长度
                w_length = len(words)
                # 为一个instance中每个char构建4个[]，分别对应BMES
                # 保存word对应的id
                gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]

                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]


                # modify
                # 表示匹配上的word以哪个char开头
                gazs_start_char_idx = [ [[] for i in range(4)] for _ in range(w_length)]
                # 表示匹配上的word以哪个char结尾
                gazs_end_char_idx = [ [[] for i in range(4)] for _ in range(w_length)]
                # 表示当前char在该word中的相对位置
                gazs_char_rel_pos = [ [[] for i in range(4)] for _ in range(w_length)]



                # 最大的B/M/E/S的长度
                max_gazlist = 0
                # 一句话中匹配到的word最大长度
                max_gazcharlen = 0
                # 对一句话中每个char
                for idx in range(w_length):
                    # 搜索以该char开头的所有words及其长度及其index
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    # 求出以固定char开头的所有words中的最大长度，并与之前的chars开头的words最大长度长度相比较
                    # 一个instance中匹配到的word的最大长度
                    if matched_length:
                        max_gazcharlen = max(max(matched_length), max_gazcharlen)

                    # 处理单个word(固定char开头的所有words)
                    # 完成gazs,gazs_count,gaz_char_Id的赋值
                    for w in range(len(matched_Id)):
                        # 保存一个word的char对应的index
                        gaz_chars = []
                        g = matched_list[w]
                        # c:一个word的一个个char
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))

                        if matched_length[w] == 1:  ## Single
                            gazs[idx][3].append(matched_Id[w])
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)

                            # 单字组成word
                            gazs_start_char_idx[idx][3].append(gaz_chars[0])
                            gazs_end_char_idx[idx][3].append(gaz_chars[0])
                            # 位置的相对编码，相对整个word的偏移量
                            gazs_char_rel_pos[idx][3].append(1)

                        # 这个word不是Single(处理该word的所有char)
                        else:
                            # 分别保存word的index,该word在数据集中出现次数,该word的char对应的index
                            gazs[idx][0].append(matched_Id[w])   ## Begin
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx][0].append(gaz_chars)

                            # char作为word的首字母
                            # last_char_idx = word_alphabet.get_index(g[-1]) # 查找word最后一个char的index
                            gazs_start_char_idx[idx][0].append(gaz_chars[0])
                            gazs_end_char_idx[idx][0].append(gaz_chars[-1])
                            gazs_char_rel_pos[idx][0].append(1)

                            # 该word的长度
                            wlen = matched_length[w]
                            gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                            gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)

                            # char作为word的末位
                            # first_char_idx = word_alphabet.get_index(g[0])
                            gazs_start_char_idx[idx+wlen-1][2].append(gaz_chars[0])
                            gazs_end_char_idx[idx+wlen-1][2].append(gaz_chars[-1])
                            gazs_char_rel_pos[idx+wlen-1][2].append(len(g))

                            for l in range(wlen-2):
                                gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)

                                # char作为word的中间字符
                                gazs_start_char_idx[idx+l+1][1].append(gaz_chars[0])
                                gazs_end_char_idx[idx+l+1][1].append(gaz_chars[-1])
                                gazs_char_rel_pos[idx+l+1][1].append(l+2)



                    for label in range(4):
                        # 某一个label为空
                        if not gazs[idx][label]:
                            # word对应的index，出现次数，对应char的index
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                            # 对空数组赋值
                            gazs_start_char_idx[idx][label].append(0)
                            gazs_end_char_idx[idx][label].append(0)
                            # 相对位置编码不能是0，需要和idx=0的char区分开来（后面为了方便统一用0）
                            gazs_char_rel_pos[idx][label].append(0)

                        # 比较BMES哪个列表长度是最长的，同时与之前的idx对应的gazs比较
                        # 得到一句话中最大长度的BMES
                        max_gazlist = max(len(gazs[idx][label]),max_gazlist)

                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    # 完成gaz_Ids的赋值，长度=instance的长度
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                # end对一句话中每个char


                # 对一句话中每个char
                ## batch_size = 1
                # 重新把以上变量做补齐操作
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []

                    for label in range(4):
                        # B/M/E/S集合大小
                        label_len = len(gazs[idx][label])
                        count_set = set(gazs_count[idx][label])
                        # print(label_len)
                        # print(count_set)

                        # 存在比该word更长的word即计数默认是0
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1]*label_len
                        # 实际长度部分用0，超出部分用1
                        mask = label_len*[0]
                        mask += (max_gazlist-label_len)*[1]

                        # 超出部分用0补齐
                        # [w_length, 4, max_gazlist]
                        gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding


                        # modify
                        gazs_start_char_idx[idx][label] += (max_gazlist-label_len)*[0]
                        gazs_end_char_idx[idx][label] += (max_gazlist-label_len)*[0]
                        gazs_char_rel_pos[idx][label] += (max_gazlist-label_len)*[0]



                        # 保存B/M/E/S任一集合中的所有padding过的char_padding
                        char_mask = []
                        # 对每个word
                        for g in range(len(gaz_char_Id[idx][label])):
                            # 最小单元word的长度
                            glen = len(gaz_char_Id[idx][label][g])
                            # 实际长度用0，超出部分用1
                            charmask = glen*[0]
                            charmask += (max_gazcharlen-glen) * [1]
                            char_mask.append(charmask)

                            # 超出部分用0补齐
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]

                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
                        # 超出部分用1
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                        # gazmask:[4, max_gazlist]
                        # [[0,0,0...,0,1,1...,1],[],[],[]]
                        gazmask.append(mask)
                        # gazcharmask:[4, max_gazlist, max_gazcharlen]
                        # [ [ [0,0,1,1],[0,0,0,1],....,[1,1,1,1],[1,1,1,1] ], [],[],[]]
                        gazcharmask.append(char_mask)
                    # [seq_len, 4, max_gazlist]
                    layergazmasks.append(gazmask)
                    # [seq_len, 4, max_gazlist, max_gazcharlen]
                    gazchar_masks.append(gazcharmask)

                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)

                # 一个instance
                instence_texts.append([words, biwords, chars, gazs, labels])
                # [instance_num]
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids, gazs_start_char_idx, gazs_end_char_idx, gazs_char_rel_pos])

            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids

# path:uni,bi,gaz; 各自的alphabet; dim=50;
# 使用词向量文件为word_alphabet构建pretrain_emb
def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    # 初始化，均匀分布
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    # 为word_alphabet中的word构建embedding
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        # 对于biword来说，大部分用不到bigram文件中的词向量
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

# uni, bi, gaz
def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # 更新embedd_dim
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

