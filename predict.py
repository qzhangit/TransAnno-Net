# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16905, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=25, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell tpye exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/qzhang/Project/scBERT/0The_lastest_Data/predata/LPS-PD_MAtrix_16905.h5ad', help='Path of data for predicting.')
parser.add_argument("--model_path", type=str, default='/home/qzhang/Project/scBERT/ckpt-5zhe-re-30isafterpanglao/finetune_best.pth', help='Path of finetuned model.')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

device = torch.device("cuda:0") 

class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad(args.data_path)
#load the label stored during the fine-tune stage
with open('label_dict', 'rb') as fp:
    label_dict = pkl.load(fp)
with open('label', 'rb') as fp:
    label = pkl.load(fp)

class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    bert_position_emb = True
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])

path = args.model_path
# print(path)
# #加载权重
# #ckpt = torch.load(path)
# model.load_state_dict(ckpt['model_state_dict'])
# for param in model.parameters():
#     param.requires_grad = False
# model = model.to(device)

# print("+++++++++++++++++++:",ckpt['state_dict'].keys())


# 加载权重
ckpt = torch.load(path)

# 创建新的state_dict，修正键名
# new_state_dict = {}
# for k, v in ckpt.items():
#     name = k.replace('module.', '')  # 去掉"module."前缀
#     new_state_dict[name] = v

# 实例化模型并加载权重
model.load_state_dict(ckpt['model_state_dict'])#跳过缺失的键,strict=False
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)



batch_size = data.shape[0] #68450
model.eval()
pred_finals = []
novel_indices = []
with torch.no_grad():
    for index in range(batch_size):
        full_seq = data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_final = pred_prob.argmax(dim=-1)
        if np.amax(np.array(pred_prob.cpu()), axis=-1) < UNASSIGN_THRES:
            novel_indices.append(index)
        pred_finals.append(pred_final.item())        
#print("这里输出=========",pred_finals)
pred_list = label_dict[pred_finals].tolist()
#pred_list = label_dict [[int(tensor.item()) for tensor in pred_finals]]
for index in novel_indices:
    pred_list[index] = 'Unassigned'
#print(pred_list)

import csv
# 打开一个文件，准备写入  
with open('predlist-afterpanglao.csv', 'w', newline='') as file:  
    writer = csv.writer(file)  
    # 将数据写入文件  
    for index, value in enumerate(pred_list):  
        writer.writerow([index, value])  

        
from datetime import datetime

current_time = datetime.now()
print("当前时间是:", current_time)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 