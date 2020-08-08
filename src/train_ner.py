from transformers import BertTokenizer, BertModel
import json
import numpy as np
import torch
import torch.nn as nn


max_seq_len = 20
BERT_ID = 'hfl/chinese-bert-wwm-ext'


def find_lcsubstr(s1, s2):
    #生成0矩阵，为方便后续计算，比字符串长度多了一列
    m=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
    mmax=0  #最长匹配的长度
    p=0 #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
            if m[i+1][j+1]>mmax:
                mmax=m[i+1][j+1]
                p=i+1
    return s1[p-mmax:p]


class DataLoader():
    def __init__(self, path, batch_size=64, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        
        with open(path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        self.questions = np.array(
            [ corpus[i]['question'] for i in range(len(corpus)) ], 
            dtype=object)
        entities = [ corpus[i]['gold_entities'] for i in range(len(corpus)) ]
        entities = [ [entity[1:-1].split('_')[0] for entity in line] \
            for line in entities ] # 有2个含有多个下划线的实体
        self.entities = np.array(entities, dtype=object)

        X, att_mask, Y = [], [], []
        for i in range(len(self.questions)):
            q = self.questions[i]
            ret = self.tokenizer(
                q, padding='max_length', 
                truncation=True, max_length=max_seq_len)

            y = [ [0] for j in range(max_seq_len) ]
            assert len(ret['input_ids']) == len(y)
            for e in self.entities[i]:
                e = find_lcsubstr(e, q)
                if e in q:
                    begin = q.index(e) + 1 # CLS
                    end = begin + len(e)
                    if end < max_seq_len-1: # SEP
                        for pos in range(begin, end):
                            y[pos] = [1]
            
            X.append(ret['input_ids'])
            att_mask.append(ret['attention_mask'])
            Y.append(y)
        
        self.X, self.att_mask = np.array(X), np.array(att_mask)
        self.Y = np.array(Y, dtype=float)
        
        self.num_data = self.X.shape[0]
        self.num_batches = int(np.ceil(self.num_data/self.batch_size))
    
    def __iter__(self):
        if self.shuffle:
            full_index = np.random.permutation(self.num_data)
        else:
            full_index = np.arange(self.num_data)
        for i in range(self.num_batches):
            index = full_index[i*self.batch_size: (i+1)*self.batch_size]
            # X         [ batch_size, max_seq_len ]
            # att_mask  [ batch_size, max_seq_len ]
            # Y         [ batch_size, max_seq_len, 1 ]
            yield self.X[index], self.att_mask[index], self.Y[index], \
                self.questions[index], self.entities[index]


class NERBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_ID)
        hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            bidirectional=True,
            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, 1)
    
    def forward(self, X, att_mask):
        # BERT
        # batch_size, seq_len && batch_size, seq_len
        ret = self.bert(X, attention_mask=att_mask)[0] 
        # batch_size, max_seq_len, hidden_size
        
        # Bi-LSTM
        lengths = att_mask.sum(dim=-1)
        lengths, indices = torch.sort(lengths, descending=True)
        ret = ret[indices] # 按长度从大到小排序
        
        ret, _ = self.lstm(
            nn.utils.rnn.pack_padded_sequence(ret, lengths, batch_first=True))
        ret, _ = nn.utils.rnn.pad_packed_sequence(ret, batch_first=True)
        
        _, indices = torch.sort(indices)
        ret = ret[indices] # 还原顺序
        # batch_size, max_seq_len, 2*hidden_size
        
        # Linear
        ret = self.fc(self.dropout(ret))
        # batch_size, max_seq_len, 1
        return ret


def restore_entities(pred, question):
    # 注意：shuffle后，pred与questions顺序不同
    all_entities = []
    for i in range(len(pred)):
        entities = []
        str = ''
        labels = pred[i][1:-1] # cls & sep
        for j in range(min(len(labels), len(question))):
            if labels[j] == 1:
                str += question[i][j]
            else:
                if len(str):
                    entities.append(str)
                    str = ''
        if len(str):
            entities.append(str)
        all_entities.append(entities)
    return all_entities


def computeF(gold, pred):
    # 注意：shuffle后，pred与gold顺序不同
    true_num, pred_num, gold_num = 0, 0, 0
    for i in range(len(gold)):
        gold_num += len(gold[i])
        pred_num += len(pred[i])
        true_num += len(set(gold[i]).intersection(set(pred[i])))
    
    try:
        precision = true_num / pred_num
        recall = true_num / gold_num
        f = 2*precision*recall / (precision+recall)
    except:
        precision = recall = f = 0.
    return precision, recall, f
    
    
def main():
    # 训练集
    train_loader = DataLoader('../data/corpus_train.json', shuffle=True)
    # 测试集
    test_loader = DataLoader('../data/corpus_test.json')
    
    # 训练
    device = torch.device('cuda:1')
    model = NERBERT().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    maxf1, stop_epoch = 0., 0
    for epoch in range(100):
        model.train()
        total_loss = 0.
        for batch in train_loader:
            X = torch.from_numpy(batch[0]).to(device)
            att_mask = torch.from_numpy(batch[1]).to(device)
            Y = torch.from_numpy(batch[2]).to(device)
            
            optimizer.zero_grad()
            pred = model(X, att_mask)
            loss = criterion(pred, Y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_pred = []
        for batch in test_loader:
            X = torch.from_numpy(batch[0]).to(device)
            att_mask = torch.from_numpy(batch[1]).to(device)
            Y = torch.from_numpy(batch[2]).to(device)
            
            pred = model(X, att_mask).tolist() # b, s, 1
            pred = [ [1 if each[0]>0.5 else 0 for each in line] \
                for line in pred ] # b, s
            all_pred.extend(pred) 
        
        pred_entities = restore_entities(all_pred, test_loader.questions)
        precision, recall, f1 = computeF(test_loader.entities, pred_entities)
        
        print('epoch %d | train loss %.4f'%(epoch+1, total_loss), end=' | ')
        print('test f1-score %.4f, precision %.4f, recall %.4f'%(
            f1, precision, precision))
        
        if f1 > maxf1:
            maxf1 = f1
            stop_epoch = 0
            torch.save(model.state_dict(), '../data/model/ner_model.pt')
        else:
            stop_epoch += 1
        if stop_epoch == 10: # early stop
            break
    
    # 测试
    model.load_state_dict(torch.load('../data/model/ner_model.pt'))
    model.eval()
    all_pred = []
    for batch in test_loader:
        X = torch.from_numpy(batch[0]).to(device)
        att_mask = torch.from_numpy(batch[1]).to(device)
        Y = torch.from_numpy(batch[2]).to(device)
        
        pred = model(X, att_mask).tolist()
        pred = [ [1 if each[0]>0.5 else 0 for each in line] \
            for line in pred ]
        all_pred.extend(pred)
    
    pred_entities = restore_entities(all_pred, test_loader.questions)
    
    precision, recall, f1 = computeF(test_loader.entities, pred_entities)
    print('precision %.4f, recall %.4f, f1 %.4f'%(precision, recall, f1))
    for i in range(200, 230):
        print(pred_entities[i])
        print(test_loader.entities[i])


if __name__ == '__main__':
    main()

