# -*- coding: utf-8 -*-
from datetime import datetime
current_time1 = datetime.now()
print("预训练开始时间是:", current_time1)
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
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16905, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=2, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-5, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using bert encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/qzhang/Project/scBERT/data/PanglaoDB_Lung_hvg_16906_MAtrix_generename.h5ad', help='Path of data for pretraining.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts-panglao_rename16906/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Pretrained model name.')

args = parser.parse_args()
local_rank = args.local_rank
rank = int(os.environ["RANK"])
is_master = rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
CLASS = args.bin_num + 2
MASK_PROB = args.mask_prob
REPLACE_PROB = args.replace_prob
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]
POS_EMBED_USING = args.pos_embed

train_losses = []
train_accuracies = []

model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
seed_all(SEED + torch.distributed.get_rank())

# 得到随机prob矩阵，True表示小于prob阈值#########生成一个与输入张量同形的 bool 张量,其中 True 表示对应位置的值小于给定概率。
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 得到不能被掩码的掩码矩阵###########生成一个与输入张量同形的 bool 张量,其中 True 表示对应位置的值在给定的 token ID 集合中
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

#根据给定的掩码概率,生成一个新的掩码张量,其中 True 表示需要被掩码的位置。
#"纯代币"是指在序列中不包含任何特殊标记或填充标记的标记或令牌
def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)  # 每个序列中平均被掩盖的数量
    num_tokens = mask.sum(dim=-1, keepdim=True)  # 每个序列除了特殊标记之外的纯代币数量
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0), mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())  # 仅允许掩盖15%的纯代币
    mask_excess = mask_excess[:, :max_masked]  # 获取15%纯代币和15%所有代币之间的差异
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)  # 使用rand(0-1)作为概率，特殊标记使用-1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)  # 获取前k个概率的索引进行掩盖
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)  # 删除掩盖不纯的差异
    new_mask = torch.zeros((batch, seq_len + 1), device=device)  # 获取(batch, seq_len)形状的零矩阵
    new_mask.scatter_(-1, sampled_indices, 1)  # 将零矩阵中的掩盖设置为1
    return new_mask[:, 1:].bool()  # 最终的掩盖，True表示掩盖

#对输入数据进行掩码操作,生成被掩码的输入张量和对应的标签张量。
def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    # 将要被忽略的掩码标记的ID添加到集合中，包括填充标记
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # 不对 [pad] 标记或其他被排除的标记（例如 [cls]、[sep]）进行掩码
    # 同样，这些特殊标记也不会被随机选择替换
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # 忽略的标记将不会被后续进行掩码
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # 获取 True/False 掩码矩阵
    # 掩码索引
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # 获取掩码的索引（掩码矩阵的非零值索引）
    # 使用概率 `replace_prob` 对输入进行掩码，被掩码的标记将以 `mask_token_id` 进行替换，其余标记保持不变
    masked_input = data.clone().detach()
    # 如果用于 MLM 的随机标记概率 > 0  #掩码语言模型（Masked Language Model, MLM）
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'#num_tokens 关键字在使用随机标记替换时必须提供
        random_token_prob = prob_mask_like(data, random_token_prob)       # 获取随机标记替换的掩码矩阵
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # 生成与输入数据相同形状的随机标记矩阵
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # 不被掩码的随机标记矩阵
        random_token_prob &= ~random_no_mask        # 获取纯掩码的随机标记矩阵
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # 随机标记替换的索引
        masked_input[random_indices] = random_tokens[random_indices]        # 将部分标记替换为随机标记
    # 对输入进行 [mask] 掩码处理
    replace_prob = prob_mask_like(data, replace_prob)     # 获取被掩码的标记的掩码矩阵
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # 获取已经被 [mask] 掩码处理的数据
    # 对于原本不会被掩码的标记，将其掩码为填充标记
    labels = data.masked_fill(~mask, pad_token_id)        # 被掩码标记的标签
    return masked_input, labels


class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0] 
        full_seq[full_seq > (CLASS - 2)] =  - 2
        full_seq = torch.from_numpy(full_seq).long()     
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad(args.data_path)
data = data.X 

data_train, data_val = train_test_split(data, test_size=0.2,random_state=SEED)
#print(data_train)

train_dataset = SCDataset(data_train)
val_dataset = SCDataset(data_val)

#print("=========val_data",val_dataset.data)

train_sampler = DistributedSampler(train_dataset)
val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# for batch in val_loader:
#     # 对批次中的数据进行操作，例如打印
#     print("val_lodar============",batch)
model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    bert_position_emb = POS_EMBED_USING
)
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# learning rate scheduler
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-7,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean').to(local_rank)
softmax = nn.Softmax(dim=-1)

dist.barrier()
for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, data in enumerate(train_loader):
        index += 1
        data = data.to(device)
        data, labels = data_mask(data)
        # print("==================data================",data)
        # print("==================datas-shape================",data.shape)
        # print("==================labels================",labels)
        # print("==================labels-shape================",labels.shape)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                # print("==================logits================",logits)
                # print("==================logits-shape================",logits.shape)
                loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
                # print("===========loss============",loss)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            # print("==================logits================",logits)
            # print("==================logits-shape================",logits.shape)
            loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
            # print("===========loss============",loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        final = softmax(logits)[..., 1:-1]
        final = final.argmax(dim=-1) + 1
        pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
        correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    # 在每个 epoch 结束后，收集训练损失和准确率
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print("train_losses: ",train_losses)
    print("train_accuracies: ",train_accuracies)
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        running_error = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, data in enumerate(val_loader):
                index += 1
                data = data.to(device)
                data, labels = data_mask(data)
                logits = model(data)
                loss = loss_fn(logits.transpose(1, 2), labels)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final = softmax(logits)[..., 1:-1]
                final = final.argmax(dim=-1) + 1
                predictions.append(final)
                truths.append(labels)
            del data, labels, logits, final
            # gather
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
            val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            val_acc = 100 * correct_num / val_num
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
    del predictions, truths

    if is_master:
        save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)




from datetime import datetime
current_time2 = datetime.now()
print("预训练结束时间是:", current_time2)