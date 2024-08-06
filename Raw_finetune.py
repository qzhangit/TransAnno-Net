# -*- coding: utf-8 -*-
from datetime import datetime
current_time1 = datetime.now()
print("训练开始时间是:", current_time1)
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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import scvi
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')#当前节点的设备id
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16905, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=6, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
#parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using bert encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/qzhang/Project/scBERT/data-all/cellblast/ALIGNED_Mus_musculus_Pancreas_Matrix.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='/home/qzhang/Project/scBERT/ckpt-all/ckpts-panglao_rename16906/panglao_pretrain_20.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpt-Pancreas/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
parser.add_argument("--resume", action="store_true", help='从最新的检查点恢复训练.')
parser.add_argument("--save_every", type=int, default=1, help='Save checkpoint every n epochs.')



args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

resume_training = args.resume

SAVE_EVERY = args.save_every  # 保存检查点的频率
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 20
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2 
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())



# 定义 SCDataset 类
class SCDataset(Dataset):
    def __init__(self, data, label=None, is_test=False):
        super().__init__()
        self.data = data
        self.label = label
        self.is_test = is_test

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)

        if not self.is_test:
            seq_label = self.label[rand_start]
            return full_seq, seq_label
        else:
            return full_seq

    def __len__(self):
        return self.data.shape[0]

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

data1 = sc.read_h5ad(args.data_path)

type_to_index = dict(zip(data1.obs['celltype'].unique(), range(len(data1.obs['celltype'].unique()))))
index_to_type = dict(zip(range(len(data1.obs['celltype'].unique())), data1.obs['celltype'].unique()))
X = data1.X
vr = data1.var
#y = [type_to_index[i] for i in list(data1.obs['celltype'])]
y = [i for i in list(data1.obs['celltype'])]
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
data = ad.AnnData(X_resampled, obs={'celltype': y_resampled}, var=vr)

label_dict_Pancreas, label_Pancreas = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored

label_names = [str(label_dict_Pancreas[i]) for i in range(len(label_dict_Pancreas))]

#store the label dict and label for prediction
with open('label_dict_Pancreas', 'wb') as fp:
    pkl.dump(label_dict_Pancreas, fp)
with open('label_Pancreas', 'wb') as fp:
    pkl.dump(label_Pancreas, fp)
print("OK")



class_num = np.unique(label_Pancreas, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label_Pancreas)
data = data.X

acc = []
f1 = []
f1w = []

    
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=SEED)

# 划分训练集为五折交叉验证的训练集和验证集
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # 在这里训练模型，并使用 X_val_fold 和 y_val_fold 进行验证
    # 每个 fold 训练出的模型可以使用 X_test 和 y_test 进行最终的测试

train_dataset = SCDataset(X_train_fold, y_train_fold)
val_dataset = SCDataset(X_val_fold, y_val_fold)
test_dataset = SCDataset(X_test, y_test)

# DataLoader 和 DistributedSampler
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
test_sampler = DistributedSampler(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)


model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    bert_position_emb = POS_EMBED_USING
)

path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
# load_model_with_adjustment(model, path)


# 遍历模型的参数，并打印参数的形状
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Shape: {param.shape}")
    
    

for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict_Pancreas.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)##交叉熵损失函数

learning_rates = []
for epoch in range(EPOCHS):
    learning_rates.append(optimizer.param_groups[0]['lr'])
    scheduler.step()
    
#print("1:\n",optimizer.param_groups)
print("2:\n",learning_rates)

#########保存损失和准确率
train_losses=[]
train_acc=[]
val_acc=[]
val_loss_list=[]
test_losses=[]
test_acc_list=[]

dist.barrier()
trigger_times = 0
max_acc = 0.0

import os
import re

def get_latest_checkpoint(ckpt_dir, model_name):
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if re.match(f'{model_name}_epoch\d+.pth', f)]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r'\d+', x).group()))
    return os.path.join(ckpt_dir, latest_checkpoint)

# 如果正在恢复训练，则加载最新的检查点
if resume_training:
    latest_checkpoint = get_latest_checkpoint(ckpt_dir, model_name)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从第 {start_epoch} 轮恢复训练.")
    else:
        print("未找到检查点。从头开始训练.")
else:
    start_epoch = 1

    
    
for i in range(start_epoch, EPOCHS+1):
    predictions = []
    truths = []
    train_loader.sampler.set_epoch(i)
    model.train()#训练模式
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        # print("=================data======",data)
        # print("========data-shape",data.shape)
        # print("========lanels",labels)
        # print("========labels-shape",labels.shape)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                # print("========logit",logits)
                # print("========logit-shape",logits.shape)
                loss = loss_fn(logits, labels)
                # print("========loss",loss)
                # print("========loss-shape",loss.shape)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        predictions.append(final)  # 添加模型预测结果
        truths.append(labels)     # 添加真实标
    # 计算训练集上的分类报告
    train_predictions = distributed_concat(torch.cat(predictions, dim=0), len(train_sampler.dataset), world_size)
    train_truths = distributed_concat(torch.cat(truths, dim=0), len(train_sampler.dataset), world_size)
    train_no_drop = train_predictions != -1
    train_predictions = np.array((train_predictions[train_no_drop]).cpu())
    train_truths = np.array((train_truths[train_no_drop]).cpu())
    
    train_f1 = f1_score(train_truths, train_predictions, average='weighted')
    train_macro_f1 = f1_score(train_truths, train_predictions, average='macro')

    epoch_loss = running_loss / index
    epoch_acc =  cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        #target_names = [str(label_dict1[i]) for i in range(len(label_dict1))]
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:.6f}  | Train_F1 Score: {train_f1:.6f} | Train_Macro_F1 Score: {train_macro_f1:.6f}  ==')
        print(confusion_matrix(train_truths, train_predictions))
        print(classification_report(train_truths, train_predictions, labels=np.arange(0, len(label_dict_Pancreas), 1), target_names=label_names, digits=4))
    ####记录损失和准确率
    train_losses.append(epoch_loss)
    train_acc.append(epoch_acc)
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()##验证模式
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
            del data_v, labels_v, logits, final_prob, final
            # gather
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions)
            #f1 = f1_score(truths, predictions, average='macro')
            val_f1 = f1_score(truths, predictions, average='weighted')
            val_macro_f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            val_loss_list.append(val_acc)
            val_acc.append(cur_acc)
            
            if is_master:
                #target_names = [str(label_dict1[i]) for i in range(len(label_dict1))]

                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {cur_acc:.6f}  | Val_F1 Score: {val_f1:.6f} | Val_Macro_F1 Score: {val_macro_f1:.6f}  ==')
            
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions,labels=np.arange(0,len(label_dict_Pancreas),1), target_names=label_names, digits=4))
                
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0  
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)#保存了在训练过程中损失最低的模型权重、优化器状态、学习率调度器状态和损失信息。文件名包含了模型名称。
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
            #save_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)#保存了当前训练 epoch 的模型权重、优化器状态、学习率调度器状态和损失信息。文件名包含了模型名称、epoch 数字。         
            #save_simple_ckpt(model,model_name, ckpt_dir)#保存了当前训练 epoch 的模型权重。文件名包含了模型名称。
            
    # 定期保存检查点
    if i % SAVE_EVERY == 0:
        save_ckpt(i, model, optimizer, scheduler, loss, model_name, ckpt_dir)
        save_simple_ckpt(model, model_name, ckpt_dir)

    del predictions, truths

# 在测试集上进行测试
model.eval()
dist.barrier()
test_running_loss = 0.0
test_predictions = []
test_truths = []
with torch.no_grad():
    for index, (data_test, labels_test) in enumerate(test_loader):
        index += 1
        data_test, labels_test = data_test.to(device), labels_test.to(device)
        logits_test = model(data_test)
        loss_test = loss_fn(logits_test, labels_test)
        test_running_loss += loss_test.item()
        softmax_test = nn.Softmax(dim=-1)
        final_prob_test = softmax_test(logits_test)
        final_test = final_prob_test.argmax(dim=-1)
        final_test[np.amax(np.array(final_prob_test.cpu()), axis=-1) < UNASSIGN_THRES] = -1
        test_predictions.append(final_test)
        test_truths.append(labels_test)
    del data_test, labels_test, logits_test, final_prob_test, final_test
    # gather
    test_predictions = distributed_concat(torch.cat(test_predictions, dim=0), len(test_sampler.dataset), world_size)
    test_truths = distributed_concat(torch.cat(test_truths, dim=0), len(test_sampler.dataset), world_size)
    no_drop_test = test_predictions != -1
    test_predictions = np.array((test_predictions[no_drop_test]).cpu())
    test_truths = np.array((test_truths[no_drop_test]).cpu())
    test_acc = accuracy_score(test_truths, test_predictions)
    test_loss = test_running_loss / index
    test_loss = get_reduced(test_loss, local_rank, 0, world_size)
    if is_master:
        #target_names = [str(label_dict1[i]) for i in range(len(label_dict1))]

        print(f'    ==  Test Loss: {test_loss:.6f} | Test Accuracy: {test_acc:.6f}  ==')
        print(confusion_matrix(test_truths, test_predictions))
        print(classification_report(test_truths, test_predictions,labels=np.arange(0,len(label_dict_Pancreas),1) ,target_names=label_names, digits=4))
    test_losses.append(test_loss)
    test_acc_list.append(test_acc)

def save_test_results(test_losses, test_acc_list, model_name, ckpt_dir):
    result_dict = {
        'test_losses': test_losses,
        'test_acc': test_acc_list
    }
    
    result_path = os.path.join(ckpt_dir, f'{model_name}_test_results.pth')
    torch.save(result_dict, result_path)
    print(f'Test results saved to: {result_path}')
# 保存测试结果
if is_master:
    save_test_results(test_losses, test_acc, model_name, ckpt_dir)
print("train_acc:",train_acc)
print("train_losses:",train_losses)
print("val_acc:",val_acc)
print("val_loss_list:",val_loss_list)
print("test_losses:",test_losses)
print("test_acc_list:",test_acc_list)



from datetime import datetime
current_time2 = datetime.now()
print("训练结束时间是:", current_time2)