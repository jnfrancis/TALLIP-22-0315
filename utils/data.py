# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetteer import Gazetteer
from utils.functions import load_pretrain_emb

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        # 对于英文来说char_alphabet是有意义的
        # 但是对于中文来说，char_alphabet和word_alphabet是一样的，因为中文数据集每一行都是一个word，同时也是char
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.gaz_count = {}
        self.gaz_split = {}
        self.biword_count = {}

        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True
        self.HP_use_count = False

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        # 使用先验知识编码
        self.use_class_emb = True
        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.posi_emb_dim = 30
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.class_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.posi_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 128
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        # 因为中文中单个word就已经是char了
        self.HP_use_char = False
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.HP_use_posi = True
        self.HP_use_context = True
        self.HP_use_global_attention = True

        self.use_boundary = True
        self.use_char_pos = True

        self.HP_num_layer = 4

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Biword alphabet size: %s"%(self.biword_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Biword embedding size: %s"%(self.biword_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))

    # train_file,dev_file,test_file构建alphabet
    # 更新word,biword,character,label对应的alphabet，以及它们对应的size
    # 最后是获得它们的tagscheme
    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        # 记录每个instance的长度
        seqlen = 0
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                # [char, label]
                pairs = line.strip().split()
                word = pairs[0]
                # 把数字换成0
                if self.number_normalized:
                    # normalize_word返回的是str
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                # 下一行还有char，组成biword
                if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                    biword = word + in_lines[idx+1].strip().split()[0]
                # 最后一个char和"-null-"组成biword
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                # biword_index = self.biword_alphabet.get_index(biword)
                # dict:key(biword);value(出现次数)
                self.biword_count[biword] = self.biword_count.get(biword,0) + 1
                for char in word:
                    self.char_alphabet.add(char)

                seqlen += 1
            # 一句话处理结束
            else:
                self.posi_alphabet_size = max(seqlen, self.posi_alphabet_size)
                seqlen = 0

        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        # 判断数据集标记策略
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
    
    # ctb.50d.vec(lexicon文件)
    # 更新self.gaz
    def build_gaz_file(self, gaz_file):
        ## build gaz file,initial read gaz embedding file
        if gaz_file:
            fins = open(gaz_file, 'r',encoding="utf-8").readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    # 构建字典树同时更新ent2type和ent2id
                    self.gaz.insert(fin, "one_source")
            print ("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print ("Gaz file is None, load nothing")

    # 把数据集中的instance和字典树相匹配
    # 更新self.gaz_alphabet(为匹配到的word构建到index的映射)
    def build_gaz_alphabet(self, input_file, count=False):
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        word_list = []
        for line in in_lines:
            # 得到一个instance的char_list表示
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            # 开始处理准备好的一句话
            else:
                # 空行上面的一个instance的长度
                w_length = len(word_list)
                # 记录所有匹配到的word(len=一个句子中所有匹配到的word)
                entitys = []
                for idx in range(w_length):
                    # 以固定char开头所匹配到的字典树中的word
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    entitys += matched_entity
                    for entity in matched_entity:
                        # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        self.gaz_alphabet.add(entity)
                        index = self.gaz_alphabet.get_index(entity)

                        # 初始化gaz_count全为0，如果查找到了，赋值还是原来的value(长度=所有匹配上的lexicon集合大小)
                        self.gaz_count[index] = self.gaz_count.get(index,0)  ## initialize gaz count

                # True
                # 依次处理entitys中的word，更新gaz_count
                if count:
                    # 把一个instance中所有匹配到的word按照长度从大到小排序
                    entitys.sort(key=lambda x: -len(x))
                    while entitys:
                        longest = entitys[0]
                        longest_index = self.gaz_alphabet.get_index(longest)
                        # 统计每个word出现次数（以整个数据集为研究对象）
                        # key:gaz_alphabet_index
                        # value:每找到一个计数+1
                        self.gaz_count[longest_index] = self.gaz_count.get(longest_index, 0) + 1

                        gazlen = len(longest)
                        # 移除子串（包括本身）
                        # 子串不计入频率计算中
                        for i in range(gazlen):
                            for j in range(i+1, gazlen+1):
                                covering_gaz = longest[i:j]
                                if covering_gaz in entitys:
                                    entitys.remove(covering_gaz)
                                    # print('remove:',covering_gaz)
                # 保存下一个instance
                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print ("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print ("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)


    def build_gaz_pretrain_emb(self, emb_path):
        print ("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,  self.gaz_emb_dim, self.norm_gaz_emb)


    #构建数据集对应的先验知识embedding
    def build_prior_knowledge_emb(self, input_file, embedding_path):
        print("build prior pretrain emb...")
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        entity_list = []
        label_list = []
        # 没有找到一个entity
        flag = 0
        temp_word = ""
        temp_label = ""
        # 找出每个instance中的所有entity
        for line in in_lines:
            # 非空行
            if len(line) > 3:
                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                if label != 'O':
                    flag = 1
                    temp_word += word
                    # 只要实体类型，不需要实体边界
                    temp_label = label.split('-')[-1]
                else:
                    # 表示之前有实体被记录下来
                    if flag == 1:
                        entity_list.append(temp_word)
                        label_list.append(temp_label)
                    # 置空方便记录下一个entity
                    temp_word = ""
                    temp_label = ""
                    flag = 0
            # 空行
            else:
                # 如果实体处于instance的最末端，也需要记录下来
                if flag == 1:
                    entity_list.append(temp_word)
                    label_list.append(temp_label)
                temp_word = ""
                temp_label = ""
                flag = 0

        # 把entity按照class分类（需要注意的是同一个entity可能会属于不同的class）
        # 需要新建一个dict(key:label;value:entity_list)
        set_label = set(label_list)
        # 初始化一个字典嵌套列表的变量
        dict_class_entity = dict()
        for i in set_label:
            dict_class_entity[i] = []

        # 相似的，新建一个字典保存每个class的占比
        dict_class_ratio = dict()

        for idx in range(len(label_list)):
            dict_class_entity[label_list[idx]].append(entity_list[idx])

        # 根据这个字典计算不同class占比
        # 首先计算每个class的entity总数
        for key in dict_class_entity.keys():
            dict_class_ratio[key] = len(dict_class_entity[key])
        # 然后计算所有entity总数
        entity_num = 0
        for value in dict_class_ratio.values():
            entity_num += value
        # 最后计算class的比例关系
        for key in dict_class_ratio.keys():
            dict_class_ratio[key] /= entity_num

        # 接下来计算class_embedding
        # 只有计算某个class内部的word出现频率和其embedding相乘再相加才是有意义的
        # 这是因为一开始频率的研究对象是整体class，而后面计算每个class的embedding时，我们需要得到的是内部class中每个word对应的频率，所以需要*class_num/

        # 一共3种方法：
        # 1. 平均池化
        # 2. 类似gaz_embed的权重的计算方法
        # 3. 聚类方法


        # 预先准备好gaz_embed
        embedd_dict = dict()
        if embedding_path != None:
            embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

        # 遍历所有class
        for key in dict_class_entity.keys():
            # 对每个entity，首先在lexicon中寻找，如果找不到则利用lstm和char_embed训练
            class_emb = np.zeros([len(dict_class_entity[key]), self.gaz_emb_dim])
            # 该class的entity总数
            class_num = len(dict_class_entity[key])
            # 遍历所有entity
            for idx in range(class_num):
                entity = dict_class_entity[key][idx]
                # 在lexicon中找到了这个word
                if self.gaz.searchId(entity) != 0:
                    class_emb[idx, :] = embedd_dict[entity]
                # 需要利用char_embed训练word_embed
                # else:+



    # 为train,dev,test，分别更新train_texts,train_Ids
    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            # HP_num_layer:4;input_file:train/dev/test_file
            self.train_texts, self.train_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz,self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet,self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





