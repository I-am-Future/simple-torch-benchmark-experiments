# ## Use Glove in Pytorch to Finish NLP task - sentiment analysis
# Presented by: Shunlin Lu   
# 
# Modified by: Lai Wei
# For assignment2 use, a template version.

# *   Wikipedia+gigaword（6B）
# *   crawler（42B）
# *   crawler（840B）
# *   twitter(27B)
# According to the size of the word embedding vector, it can be divided into different dimensions such as 50 dims, 100 dims, 200 dims, etc.

#
import os
import sys
import time
import random
import datetime

import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchtext
from torchtext.vocab import GloVe 
from torchtext.data import get_tokenizer

import config_utils
from clean_utils import expand_contractions, clean_texts
from lr_scheduler_utils import CosineAnnealingLRWarmup

# ### Set random seed

seed = 114514
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ### Define Hyper-parameters

configs = {
    'work_dir': '.', 
    'device': 'cuda:5', # we have 8 GPUs!
    'batch_size': 16, 
    'train_ratio': 0.7, 
    'optimizer_config': {
        'lr': 5e-4, 
        # 'weight_decay': 1e-6, 
    }, 
    'scheduler_config': {
        'T_max': 100,
        'eta_min': 5e-5,
        'last_epoch': -1,
        'warmup_steps': 5,
        'warmup_start_lr': 1e-5
    }, 
    'total_epoch': 20, 
    'glove_name': '6B', 
    'glove_dim': 200,
    'model_config': {
        'hidden_units': 200, 
        'num_layers': 4, 
        'dropout_rate': 0.25
    }, 
    'enable_tb': False
}

# build the CONFIG object
configs = config_utils.CONFIG(configs)

# ### Define datasets

GLOVE = GloVe(name=configs['glove_name'], dim=configs['glove_dim'])

class TweetDataset(Dataset):

    def __init__(self, fname: str, is_train: bool = True) -> None:
        ''' A dataset object to read tweets.
            @input fname: the .csv file name.
            @input is_train: True if is training dataset, else the testing dataset.
        '''
        super().__init__()
        self.tokenizer = get_tokenizer('basic_english')
        self.is_train = is_train
        # read in the data
        df = pd.read_csv(fname)
        # preprocessing the data
        df['keyword'] = df['keyword'].fillna('unknown')
        df['text'] = df['text'] + ' ' + df['keyword']
        df['text'] = df['text'].apply(expand_contractions)       
        df['text'] = df['text'].apply(clean_texts)  
        # make the data in a list, for later use.
        self.data = [] # all-in-a-list. (id,keyword,location,text(embedding),target)
        for i in range(len(df)):
            if is_train:
                self.data.append((
                    df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], 
                    GLOVE.get_vecs_by_tokens(self.tokenizer(df.iloc[i, 3])), 
                    torch.tensor(df.iloc[i, 4], dtype=torch.int64)
                ))
            else: 
                self.data.append((
                    df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], 
                    GLOVE.get_vecs_by_tokens(self.tokenizer(df.iloc[i, 3])), 
                ))
        print('Preparation completed! Total:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        if self.is_train:
            return item[0], item[3], item[4]
        return item[0], item[3]

# collate functions
def collate_fn_train(batch):
    id, x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    y = torch.Tensor(y)#.long()
    return id, x_pad, y

def collate_fn_test(batch):
    id, x = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return id, x_pad

train_set = TweetDataset('./train.csv')
test_set = TweetDataset('./test.csv', False)

train_size = int(len(train_set) * configs['train_ratio'])
val_size = len(train_set) - train_size

train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

train_dataloader = DataLoader(train_set,
                batch_size=configs['batch_size'],
                shuffle=True,
                num_workers=2, 
                collate_fn=collate_fn_train)
val_dataloader = DataLoader(val_set,
                batch_size=configs['batch_size'],
                shuffle=True,
                num_workers=2, 
                collate_fn=collate_fn_train)
test_dataloader = DataLoader(test_set,
                batch_size=configs['batch_size'],
                shuffle=False,
                num_workers=2, 
                collate_fn=collate_fn_test)


# ### Define model

class TransformerModel(torch.nn.Module):
    def __init__(self, hidden_units, num_layers, dropout_rate):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, 
                                                   nhead=8, dropout=dropout_rate, batch_first=True)
        self.core = nn.TransformerEncoder(encoder_layer, num_layers)
        self.Sigmoid = nn.Sigmoid()

        self.linear2 = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor):
        # x shape: [batch, max_word_length, embedding_length]
        # emb = self.drop(x)
        output = self.core(x)
        output = output[:, -1]
        output = self.linear2(output)
        output = self.Sigmoid(output)
        return output

model = TransformerModel(**configs['model_config']).to(configs['device'])

# ### Training!
# begin training now!!!
configs.log_string(f'Total parameters: {config_utils.count_params(model)}')
configs.log_string('Begin training now...')

optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer_config'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, 0.6)
scheduler = CosineAnnealingLRWarmup(optimizer, **configs['scheduler_config'])
citerion = torch.nn.BCELoss()

total_epoch = configs['total_epoch']
best_acc = float('-inf')

for epoch in range(total_epoch):
    epoch_start_time = datetime.datetime.now()

    # train!
    model.train()
    train_loss_sum = 0
    train_correct = 0
    for _, x, y in train_dataloader: # the first one is "id", which is no use in training!
        x, y = x.to(configs['device']), y.to(configs['device'])

        y_hat = model(x).squeeze(-1)
        loss = citerion(y_hat, y)

        train_correct += ((y_hat > 0.5) == y).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        train_loss_sum += loss.item() * configs['batch_size']
    scheduler.step()

    # eval!
    model.eval()
    val_loss_sum = 0
    val_correct = 0
    with torch.no_grad():
        for _, x, y in val_dataloader: # the first one is "id", which is no use in validation!
            x, y = x.to(configs['device']), y.to(configs['device'])

            y_hat = model(x).squeeze(-1)
            loss = citerion(y_hat, y)

            val_correct += ((y_hat > 0.5) == y).sum()
            val_loss_sum += loss.item() * configs['batch_size']

    # logging!
    train_loss = train_loss_sum / len(train_set)
    train_acc = train_correct / len(train_set)
    val_loss = val_loss_sum / len(val_set)
    val_acc = val_correct / len(val_set)
    
    pt_path = os.path.join(configs["work_dir"], 'last.pth')
    torch.save(model.state_dict(), pt_path)

    epoch_end_time = datetime.datetime.now()
    duration = (epoch_end_time - epoch_start_time).seconds

    if val_acc > best_acc:
        best_acc = val_acc # I added here.
        pt_path = os.path.join(configs["work_dir"], 'best.pth')
        torch.save(model.state_dict(), pt_path)
        configs.log_string(f'{time.ctime()}: epoch {epoch+1}/{total_epoch}, duration={duration} train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, saving model.')
    else:
        configs.log_string(f'{time.ctime()}: epoch {epoch+1}/{total_epoch}, duration={duration} train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, ')
    
