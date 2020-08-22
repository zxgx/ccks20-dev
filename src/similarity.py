import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


max_seq_len = 55
BERT_ID = 'hfl/chinese-bert-wwm-ext'

class DataLoader():
    def __init__(self, path, batch_size=64, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        
        X, Y, type_ids, att_mask = [], [], [], []
        df = pd.read_csv(path, encoding='utf-8')
        
        for i in range(df.shape[0]):
            s1, s2, label = df.iloc[i]
            ret = self.tokenizer(
                s1, s2, padding='max_length', 
                truncation='longest_first', max_length=max_seq_len
            )
            
            X.append(ret['input_ids'])
            Y.append(label)
            type_ids.append(ret['token_type_ids'])
            att_mask.append(ret['attention_mask'])
        self.X, self.Y = np.array(X, dtype=np.int), np.array(Y, dtype=np.int)
        self.type_ids = np.array(type_ids, dtype=np.int) 
        self.att_mask = np.array(att_mask, dtype=np.int)
        
        self.num_data = self.X.shape[0]
        self.num_batch = int(np.ceil(self.num_data/batch_size))
    
    def __iter__(self):
        if self.shuffle:
            full_range = np.random.permutation(self.num_data)
        else:
            full_range = np.arange(self.num_data)
        for i in range(self.num_batch):
            this_index = full_range[i*self.batch_size: (i+1)*self.batch_size]
            # X [ num_data, max_seq_len ]
            # Y [ num_data ]
            # type_ids [ num_data, max_seq_len ]
            # att_mask [ num_data, max_seq_len ]
            yield self.X[this_index], self.Y[this_index], \
                self.type_ids[this_index], self.att_mask[this_index]


class BertSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_ID)
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x, type_ids, att_mask):
        ret = self.bert(
            input_ids=x, attention_mask=att_mask, token_type_ids=type_ids)
        
        # 原代码采用pool句子表示，即[cls] -> fc -> tanh的输出
        ret = self.fc(ret[1])
        
        # [ batch_size, max_seq_len, hidden_size ]
        # ret, _ = torch.max(ret[0], dim=1) 
        # [ batch_size, hidden_size ]
        # ret = self.fc(ret)
        # [ batch_size, 2 ]
        return ret


def train():
    train_loader = DataLoader('data/train.csv', shuffle=True)
    dev_loader = DataLoader('data/dev.csv')
    
    device = torch.device('cuda:1')
    model = BertSim().to(device)
    
    criterion = nn.CrossEntropyLoss()
    # 原代码有warmup等
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
    maxf1, stop_epoch = 0., 0
    for epoch in range(100):
        model.train()
        tot_loss = 0.
        for batch in train_loader:
            x = torch.from_numpy(batch[0]).to(device)
            y = torch.from_numpy(batch[1]).to(device)
            type_ids = torch.from_numpy(batch[2]).to(device)
            att_mask = torch.from_numpy(batch[3]).to(device)
            
            optimizer.zero_grad()
            logits = model(x, type_ids, att_mask)
            loss = criterion(logits, y)
            tot_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            
        acc, f1, ff1 = evaluate(model, dev_loader, device)
        print('epoch: %d, train loss: %.4f, dev acc: %.4f, dev f1: %.4f, my f1: %.4f'%\
            (epoch, tot_loss, acc, f1, ff1))
        if f1 > maxf1:
            maxf1, stop_epoch = f1, 0
            torch.save(model.state_dict(), '../data/model/similarity.pt')
        else:
            stop_epoch += 1
        if stop_epoch == 10:
            print('Stop training at epoch %d'%(epoch-10))
            break


def evaluate(model, data_loader, device):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            x = torch.from_numpy(batch[0]).to(device)
            type_ids = torch.from_numpy(batch[2]).to(device)
            att_mask = torch.from_numpy(batch[3]).to(device)
            
            logits = model(x, type_ids, att_mask)
            
            pred.append(logits.argmax(1).cpu().numpy())
            gold.append(batch[1])
    pred, gold = np.hstack(pred), np.hstack(gold)
    print(pred.tolist()[:15], '\n', gold.tolist()[:15], '\n')
    f1 = metrics.f1_score(gold.tolist(), pred.tolist())
    ff1 = f1_score(gold.tolist(), pred.tolist())
    acc = (pred==gold).sum() / data_loader.num_data
    return acc, f1, ff1


def f1_score(gold, pred):
    assert len(gold) == len(pred), 'prediction unmatches with ground truth'
    gold_true, pred_true, tp = 0, 0, 0
    for g, p in zip(gold, pred):
        if g == 1:
            gold_true += 1
        if p == 1:
            pred_true += 1
        if g == 1 and p == 1:
            tp += 1
    precision = tp / pred_true
    recall = tp / gold_true
    f1 = 2*precision*recall / (precision+recall)
    return f1


def predict(model, tokenizer, device, s1, s2):
    ret = tokenizer(
        s1, s2, padding='max_length', 
        truncation='longest_first', max_length=max_seq_len
    )
    ids = torch.tensor([ret['input_ids']], dtype=torch.long).to(device)
    type_ids = torch.tensor([ret['token_type_ids']], dtype=torch.long).to(device)
    att_mask = torch.tensor([ret['attention_mask']], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(ids, type_ids, att_mask)
    return logits


if __name__ == '__main__':
    train()

    model = BertSim()
    device = torch.device('cuda:1')
    model.load_state_dict(torch.load('../data/model/similarity.pt'))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(BERT_ID)
    s1 = '借了钱，但还没有通过，可以取消吗？'
    s2 = '可否取消'
    s3 = '一天利息好多钱'
    s4 = '1万利息一天是5元是吗'
    ret = predict(model, tokenizer, device, s1, s2)
    print(ret.argmax(1))
    ret = predict(model, tokenizer, device, s3, s4)
    print(ret.argmax(1))

