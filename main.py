# -*- coding: utf-8 -*-
import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import matplotlib.pyplot as plt

from utils.metric import get_ner_fmeasure
from utils.data import Data
from model.gazlstm import GazLSTM as SeqModel

# 在导入matplotlib的时候指定不需要GUI的backend
# plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


# 控制台输出写入到log文件中
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    # 为数据集中所有的word=char,biword,label构建对应的alphabet(content:index)
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    # 构建了字典树（ctb.50d.vec）
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file,count=True)
    data.build_gaz_alphabet(dev_file,count=True)
    data.build_gaz_alphabet(test_file,count=True)
    data.fix_alphabet()
    return data

# 逐个元素比较
def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    # 这里的*是逐个元素操作
    right_token = np.sum(overlaped * mask)
    # true_seq_len
    total_token = mask.sum()

    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    # GPU转换到CPU上
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        # 把index转换为text，利用alphabet
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]

        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def print_batchword(data, batch_word, n):
    with open("labels/batchwords.txt", "a") as fp:
        for i in range(len(batch_word)):
            words = []
            for id in batch_word[i]:
                words.append(data.word_alphabet.get_instance(id))
            fp.write(str(words))

def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print( "Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print( "Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

# 设置学习率衰减
def lr_decay(optimizer, epoch, decay_rate, init_lr, logger):
    lr = init_lr * ((1-decay_rate)**epoch)
    logger.info("Learning rate is setted as:" + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print( "Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    # [batch_size, seq_len]
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    gazes = []
    for batch_id in range(total_batch):
        with torch.no_grad():
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end =  train_num
            instance = instances[start:end]
            if not instance:
                continue
            gaz_list,batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos  = batchify_with_label(instance, data.HP_gpu, data.HP_num_layer, True)
            tag_seq, gaz_match = model(gaz_list,batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos)

            gaz_list = [data.gaz_alphabet.get_instance(id) for batchlist in gaz_match if len(batchlist)>0 for id in batchlist ]
            gazes.append( gaz_list)

            if name == "dev":
                # 从index转换到text
                pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
            else:
                pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
            pred_results += pred_label
            gold_results += gold_label
    decode_time = time.time() - start_time
    # 每秒钟可以推断多少条instance
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results, gazes


def get_text_input(self, caption):
    caption_tokens = self.tokenizer.tokenize(caption)
    caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
    caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
    if len(caption_ids) >= self.max_seq_len:
        caption_ids = caption_ids[:self.max_seq_len]
    else:
        caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
    caption = torch.tensor(caption_ids)
    return caption

# batch_size个instance(train_Ids), gpu=True, num_layer=4
# 把train_Ids的列表格式转换成可训练的tensor变量
def batchify_with_label(input_batch_list, gpu, num_layer, volatile_flag=False):

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    # 这里[2]char_Ids没有意义所以没有写
    # [w_length, 4, max_gazlist]
    # [w_length, 2, matched_Id_length]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    # batch_size*[seq_len, 4, max_gazlist]   补齐部分用0替代([w_length, 4, max_gazlist])
    layer_gazs = [sent[5] for sent in input_batch_list]
    # 同上
    gaz_count = [sent[6] for sent in input_batch_list]
    # batch_size*[seq_len, 4, max_gazlist, max_gazchalen]
    gaz_chars = [sent[7] for sent in input_batch_list]
    # batch_size * [seq_len, 4, max_gazlist]
    gaz_mask = [sent[8] for sent in input_batch_list]
    # batch_size * [seq_len, 4, max_gazlist, max_gazcharlen]
    gazchar_mask = [sent[9] for sent in input_batch_list]
    ### bert tokens
    bert_ids = [sent[10] for sent in input_batch_list]


    # modify
    # [batch_size, seq_len, 4, max_gazlist]
    gazs_start_char_idx = [sent[11] for sent in input_batch_list]
    gazs_end_char_idx = [sent[12] for sent in input_batch_list]
    gazs_char_rel_pos = [sent[13] for sent in input_batch_list]



    # 每个instance长度
    # 之前的train_Ids是按照一个Instance内部的max_gazlist和max_gazcharlen来计算的，这次要考虑到所有batch
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()

    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    ### bert seq tensor
    bert_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()
    bert_mask = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()

    # 一个char的B集合对应的lexicon大小(每个instance都已经做了补齐操作，只不过max_gazlist长度各自不一样)
    # 第i个instance的第一个char的B集合
    gaz_num = [len(layer_gazs[i][0][0]) for i in range(batch_size)]
    # 相当于求所有instance的max_gazlist中的max
    max_gaz_num = max(gaz_num)
    layer_gaz_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()
    gaz_count_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).float()
    # 一个instance中匹配到的最大长度
    # 第i个instance中的第1个char的B集合的第一个word长度，由于一个Instance中的word长度一样，故可以只看[0][0][0]
    gaz_len = [len(gaz_chars[i][0][0][0]) for i in range(batch_size)]
    max_gaz_len = max(gaz_len)
    gaz_chars_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).long()
    gaz_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num).byte()
    gazchar_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).byte()


    # modify
    gazs_start_charidx_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()
    gazs_end_charidx_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()
    gazs_char_relpos_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()


    # b表示的是index
    # seq, biseq/biwords, label, mask, bert_mask, bert_id: [seq_len]
    # layergaz: [seq_len, 4, max_gazlist]
    # gazmask, gazcount: [seq_len, 4, max_gazlist]
    # gazchar, gazchar_mask: [seq_len, 4, max_gazlist, max_gazchalen]
    for b, (seq, bert_id, biseq, label, seqlen, layergaz, gazmask, gazcount, gazchar, gazchar_mask, gaznum, gazlen, gazfiridx, gazlasidx, gazrelpos) in enumerate(zip(words, bert_ids, biwords, labels, word_seq_lengths, layer_gazs, gaz_mask, gaz_count, gaz_chars, gazchar_mask, gaz_num, gaz_len, gazs_start_char_idx, gazs_end_char_idx, gazs_char_rel_pos)):
        # seqlen:每个instance实际长度
        word_seq_tensor[b, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[b, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[b, :seqlen] = torch.LongTensor(label)
        layer_gaz_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(layergaz)
        # 实际句子长度
        mask[b, :seqlen] = torch.Tensor([1]*int(seqlen))
        bert_mask[b, :seqlen+2] = torch.LongTensor([1]*int(seqlen+2))
        gaz_mask_tensor[b, :seqlen, :, :gaznum] = torch.ByteTensor(gazmask)
        gaz_count_tensor[b, :seqlen, :, :gaznum] = torch.FloatTensor(gazcount)
        gaz_count_tensor[b, seqlen:] = 1
        gaz_chars_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.LongTensor(gazchar)
        gazchar_mask_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.ByteTensor(gazchar_mask)

        ##bert
        bert_seq_tensor[b, :seqlen+2] = torch.LongTensor(bert_id)

        # modify
        gazs_start_charidx_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(gazfiridx)
        gazs_end_charidx_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(gazlasidx)
        gazs_char_relpos_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(gazrelpos)


    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        layer_gaz_tensor = layer_gaz_tensor.cuda()
        gaz_chars_tensor = gaz_chars_tensor.cuda()
        gaz_mask_tensor = gaz_mask_tensor.cuda()
        gazchar_mask_tensor = gazchar_mask_tensor.cuda()
        gaz_count_tensor = gaz_count_tensor.cuda()
        mask = mask.cuda()
        bert_seq_tensor = bert_seq_tensor.cuda()
        bert_mask = bert_mask.cuda()

        # modify
        gazs_start_charidx_tensor = gazs_start_charidx_tensor.cuda()
        gazs_end_charidx_tensor = gazs_end_charidx_tensor.cuda()
        gazs_char_relpos_tensor = gazs_char_relpos_tensor.cuda()

    # print(bert_seq_tensor.type())
    return gazs, word_seq_tensor, biword_seq_tensor, word_seq_lengths, label_seq_tensor, layer_gaz_tensor, gaz_count_tensor,gaz_chars_tensor, gaz_mask_tensor, gazchar_mask_tensor, mask, bert_seq_tensor, bert_mask, gazs_start_charidx_tensor, gazs_end_charidx_tensor, gazs_char_relpos_tensor



def train(data, save_model_dir, seg=True, pic_file=None, logger=None):

    print("=========Training model=========")
    # lstm
    print("Training with {} model.".format(data.model_type))

    #data.show_data_summary()

    print("======Initialize SeqModel======")
    model = SeqModel(data)
    print("======finished built model======")
    print("======print model structure======")
    print(model)
    print("======End======")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters, lr=data.HP_lr)

    # 在验证集上最好的结果
    best_dev = -1
    best_dev_p = -1
    best_dev_r = -1

    best_test = -1
    best_test_p = -1
    best_test_r = -1

    # 记录最好的结果在第几轮epoch出现
    best_idx = -1

    # 记录一轮epoch得到的损失值，绘制loss下降曲线图
    loss_epoch = []
    loss_average = []


    ## start training
    logger.info("\n=========打印训练过程的结果=========")
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        logger.info("Epoch: %s/%s" %(idx,data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr, logger)
        instance_count = 0
        # end==500时，统计的loss
        sample_loss = 0
        # 用于反向传播，梯度下降
        batch_loss = 0
        # epoch的所有loss
        total_loss = 0
        right_token = 0
        whole_token = 0
        # 打乱instance顺序
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            # 没有意义，因为train_Ids已经打乱了
            words = data.train_texts[start:end]
            # 空，重新进入循环
            if not instance:
                continue

            gaz_list,  batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos = batchify_with_label(instance, data.HP_gpu,data.HP_num_layer)

            instance_count += 1
            # 预测label和真实label得到的loss；预测label得到的tag_seq
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask, gaz_first_idx, gaz_last_idx, gaz_rel_pos)

            loss_epoch.append(loss.item())

            # 参数维度：[batch_size, seq_len]
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                # 这里的acc是通过每个label的逐一对比计算出来的
                logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                sys.stdout.flush()
                sample_loss = 0
            # 每一个epoch，进行一次梯度下降，更新学习率
            if end%data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        # 对最后一个total_batch结束后再进行一次acc的计算，同上输出
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
        # epoch结束后
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        # 每秒钟可以推断出多少个instance
        logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))


        logger.info("start dev and test!")
        # 验证集上的评估
        speed, acc, p, r, f, pred_labels, gazs = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        # seg=True
        if seg:
            current_score = f
            logger.info("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            logger.info("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            if seg:
                logger.info( "Exceed previous best f score:" + str(best_dev))

            else:
                logger.info( "Exceed previous best acc score:" + str(best_dev))

            model_name = save_model_dir + "/" + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            #best_dev = current_score
            best_dev_p = p
            best_dev_r = r


        # ## decode test（在测试集上进行验证）
        speed, acc, p, r, f, pred_labels, gazs = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            current_test_score = f
            logger.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            current_test_score = acc
            logger.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))


        # 验证集上的F值提升了
        if current_score > best_dev:
            best_dev = current_score
            best_test = current_test_score
            best_test_p = p
            best_test_r = r

        logger.info("Best dev score: p:{}, r:{}, f:{}".format(best_dev_p,best_dev_r,best_dev))
        logger.info("Test score: p:{}, r:{}, f:{}".format(best_test_p,best_test_r,best_test))

        logger.info("End\n")

        loss_mean = np.mean(loss_epoch)
        loss_average.append(loss_mean)

        gc.collect()


    logger.info("=========End打印训练结果=========")

    # 训练结束，绘制loss下降曲线
    x_epoch = range(data.HP_iteration)
    # print(x_epoch)
    # print(loss_average)
    plt.plot(x_epoch, loss_average, color="r", label="loss")

    # 设置x轴，y轴，标题，label
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss-epoch 曲线图")
    plt.legend()

    # 保存同时显示
    plt.savefig(pic_file)
    plt.show()

    # data.result_file = "result/result.txt"
    # 把最好的结果（在验证集和测试集上）写入文件中保存起来
    with open(data.result_file,"a") as f:
        f.write(save_model_dir+'\n')
        f.write("Best dev score: p:{}, r:{}, f:{}\n".format(best_dev_p,best_dev_r,best_dev))
        f.write("Test score: p:{}, r:{}, f:{}\n\n".format(best_test_p,best_test_r,best_test))
        f.close()

# 保存的模型地址，Data类，test, true, true
def load_model_decode(model_dir, data, name, gpu, seg=True, logger=None):
    data.HP_gpu = gpu
    logger.info( "Load Model from file: ", model_dir)
    model = SeqModel(data)

    model.load_state_dict(torch.load(model_dir))

    logger.info("Decode %s data ..."%(name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, gazs = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        logger.info("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        logger.info("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))

    return pred_results


def print_results(pred, modelname=""):
    toprint = []
    for sen in pred:
        sen = " ".join(sen) + '\n'
        toprint.append(sen)
    with open(modelname+'_labels.txt','w') as f:
        f.writelines(toprint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding',  help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument('--modelpath', default="results/save_models/")
    parser.add_argument('--modelname', default="/model")
    parser.add_argument('--dataset', choices=['resume', 'weibo', 'ontonotes4', 'msra'], help='Name of Dataset', default="resume")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="results/save_dset/")
    parser.add_argument('--train', default="data/ResumeNER/train.char.bmes")
    parser.add_argument('--dev', default="data/ResumeNER/dev.char.bmes" )
    parser.add_argument('--test', default="data/ResumeNER/test.char.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw', help='writen by myself', default="data/raw_file/demo.char.bmes")
    parser.add_argument('--output', help='Dir of save raw_file results', default="results/raw_file_results/demo_predict.result")
    parser.add_argument('--seed',default=1023,type=int)
    parser.add_argument('--labelcomment', default="")
    parser.add_argument('--resultfile',default="results/best_results/")
    parser.add_argument('--logfile', help='Dir of saved console output results', default="results/logs/console_training_metric.log")
    parser.add_argument('--picfile', help='Dir of saved loss-epoch pic', default="results/pic")
    parser.add_argument('--num_iter',default=80,type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--model_type', default='lstm')
    parser.add_argument('--drop', type=float, default=0.5)

    parser.add_argument('--use_biword', dest='use_biword', action='store_true', default=True) # 原始False
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_char', dest='use_char', action='store_true', default=False) # 原始False
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_count', action='store_true', default=True)
    parser.add_argument('--use_bert', action='store_true', default=False) # 原始False

    # 使用全局信息
    parser.add_argument('--use_context', action='store_true', default=True)
    parser.add_argument('--use_posi_emb', action='store_true', default=True)
    parser.add_argument('--use_global_attention', action='store_true', default=True)

    # parser.add_argument('--use_class_emb', help='encode prior knowledge', default=False)

    # 使用边界和char位置信息
    parser.add_argument('--use_boundary', action='store_true', default=True)
    parser.add_argument('--use_char_pos', action='store_true', default=False)

    args = parser.parse_args()



    # 更改部分args的值
    if args.dataset != 'weibo':
        args.train = "data/" + args.dataset + "/train.char.bmes"
        args.dev = "data/" + args.dataset + "/dev.char.bmes"
        args.test = "data/" + args.dataset + "/test.char.bmes"
    else:
        args.train = "data/" + args.dataset + "/train.all.bmes"
        args.dev = "data/" + args.dataset + "/dev.all.bmes"
        args.test = "data/" + args.dataset + "/test.all.bmes"
    args.savedset = args.savedset + "save_" + args.dataset + ".dset"
    args.modelname = args.modelpath + args.dataset # + args.modelname
    args.resultfile = args.resultfile + "result_" + args.dataset + ".txt"
    args.logfile = "results/logs/" + args.dataset + "/console_training_metric.log"
    args.picfile = "results/pics/loss_epoch_" + args.dataset + ".png"


    seed_num = args.seed
    set_seed(seed_num)

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    # model_dir = args.loadmodel
    output_file = args.output
    # str--->bool（True）
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    # train/test
    status = args.status.lower()

    # loss_epoch曲线图
    pic_file = args.picfile

    # 模型保存路径
    save_model_dir = args.modelname
    # 数据集保存路径
    save_data_name = args.savedset
    # GPU是否可用
    gpu = torch.cuda.is_available()

    # char_emb = "../CNNNERmodel/data/gigaword_chn.all.a2b.uni.ite50.vec"
    # bichar_emb = "../CNNNERmodel/data/gigaword_chn.all.a2b.bi.ite50.vec"
    # gaz_file = "../CNNNERmodel/data/ctb.50d.vec"

    char_emb = "./data/embedding/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "./data/embedding/gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = "./data/embedding/ctb.50d.vec"

    sys.stdout.flush()

    if status == 'train':
        # 数据集文件是否存在，存在则直接加载
        if os.path.exists(save_data_name):
            print('Loading processed data')
            with open(save_data_name, 'rb') as fp:
                data = pickle.load(fp)
            data.HP_num_layer = args.num_layer
            # 为了公平地比较，batch_size设置为1
            data.HP_batch_size = args.batch_size
            data.HP_iteration = args.num_iter
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.use_bigram = args.use_biword
            data.HP_use_char = args.use_char
            data.HP_hidden_dim = args.hidden_dim
            data.HP_dropout = args.drop
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
            # data.use_class_emb = args.use_class_emb
            data.HP_use_posi = args.use_posi_emb
            data.HP_use_context = args.use_context
            data.HP_use_global_attention = args.use_global_attention

            data.use_boundary = args.use_boundary
            data.use_char_pos = args.use_char_pos

        # 生成数据集文件save.dset
        else:
            data = Data()
            data.HP_gpu = gpu
            data.HP_use_char = args.use_char
            data.HP_batch_size = args.batch_size
            data.HP_num_layer = args.num_layer
            data.HP_iteration = args.num_iter
            data.use_bigram = args.use_biword
            data.HP_dropout = args.drop
            data.norm_gaz_emb = False
            data.HP_fix_gaz_emb = False
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.HP_hidden_dim = args.hidden_dim
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
            # data.use_class_emb = args.use_class_emb
            data.HP_use_posi = args.use_posi_emb
            data.HP_use_context = args.use_context
            data.HP_use_global_attention = args.use_global_attention

            data.use_boundary = args.use_boundary
            data.use_char_pos = args.use_char_pos

            # Data类, ctb.50d.vec, 训练集;验证集;测试集
            data_initialization(data, gaz_file, train_file, dev_file, test_file)
            data.generate_instance_with_gaz(train_file,'train')
            data.generate_instance_with_gaz(dev_file,'dev')
            data.generate_instance_with_gaz(test_file,'test')
            data.build_word_pretrain_emb(char_emb)
            data.build_biword_pretrain_emb(bichar_emb)
            data.build_gaz_pretrain_emb(gaz_file)
            # 构建class_embedding
            data.build_prior_knowledge_emb(train_file, gaz_file)

            print('Dumping data')
            with open(save_data_name, 'wb') as f:
                pickle.dump(data, f)
            set_seed(seed_num)
        print('data.use_biword=',data.use_bigram)
        # Data类，model_dir, true
        # 输出各种变量
        data.show_data_summary()

        log_file = args.logfile
        logger = get_logger(args.logfile)
        logger.info(save_data_name.split("/")[-1])

        # 把当前实验设置写进Log文件中
        logger.info("use_biword:"+str(args.use_biword))
        logger.info("use_count:"+str(args.use_count))
        logger.info("use_bert:"+str(args.use_bert))
        logger.info("use_context:"+str(args.use_context))
        logger.info("use_global_attention:"+str(args.use_global_attention))
        logger.info("use_boundary:"+str(args.use_boundary))
        logger.info("use_char_pos:"+str(args.use_char_pos))
        logger.info(str(args.model_type))

        train(data, save_model_dir, seg, pic_file, logger)

    elif status == 'test':

        test_log_file = args.logfile + "/console_testing_metric.log"
        logger = get_logger(test_log_file)
        logger.info(save_data_name.split("/")[-1])

        print('Loading processed data')
        with open(save_data_name, 'rb') as fp:
            data = pickle.load(fp)
        data.HP_num_layer = args.num_layer
        data.HP_iteration = args.num_iter
        data.label_comment = args.labelcomment
        data.result_file = args.resultfile
        # data.HP_use_gaz = args.use_gaz
        data.HP_lr = args.lr
        data.use_bigram = args.use_biword
        data.HP_use_char = args.use_char
        data.model_type = args.model_type
        data.HP_hidden_dim = args.hidden_dim
        data.HP_use_count = args.use_count
        data.generate_instance_with_gaz(test_file,'test')

        logger.info("test")

        load_model_decode(save_model_dir, data, 'test', gpu, seg, logger)

    else:
        print( "Invalid argument! Please use valid arguments! (train/test/decode)")




