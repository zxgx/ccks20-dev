import time
import jieba
from transformers import BertTokenizer, BertModel
import torch
import json

from train_ner import max_seq_len, BERT_ID, NERBERT


class MentionExtractor():
    def __init__(self):
        # 分词词典
        self.segment_dict = dict()
        with open('../data/segment_dict.txt', 'r', encoding='utf-8') as f:
            for line in f:
                self.segment_dict[line.strip()] = 0
        
        begin = time.time()
        jieba.load_userdict('../data/segment_dict.txt')
        print('加载用户分词词典时间为:%.2fs'%(time.time()-begin))
        
        # NERBERT tokenizer & model
        self.tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        self.device = torch.device('cuda:0')
        self.ner_model = NERBERT()
        self.ner_model.load_state_dict(
            torch.load('../data/model/ner_model.pt'))
        self.ner_model.to(self.device)
        print("mention extractor loaded")
    
    def get_entity_mention(self, corpus, verbose=False):
        for i in range(len(corpus)):
            item = corpus[i]
            q = item['question']
            item['entity_mention'] = self.extract_mentions(q)
            corpus[i] = item
            if verbose:
                print(q)
                print(item['entity_mention'])
        return corpus
    
    def extract_mentions(self, q):
        # {mention - mention}
        entity_mention = {}
        
        # jieba分词
        mentions = []
        tokens = jieba.lcut(q)
        for t in tokens:
            if t in self.segment_dict:
                mentions.append(t)
        # print(mentions)

        # NERBERT
        ret = self.tokenizer(
            q, padding='max_length', truncation=True, max_length=max_seq_len)
        X, att_mask = ret['input_ids'], ret['attention_mask']
        
        X = torch.tensor([X], dtype=torch.long).to(self.device) 
        att_mask = torch.tensor([att_mask], dtype=torch.long).to(self.device)
        # 1, max_seq_len && 1, max_seq_len
        pred = self.ner_model(X, att_mask)
        # 1, max_seq_len, 1
        assert pred.shape[0] == pred.shape[2]
        pred = pred.squeeze(0).squeeze(1).tolist()
        # max_seq_len
        pred = [ 1 if each>0.5 else 0 for each in pred ]
        
        bert_mentions = self.restore_entities(pred, q)
        # print(bert_mentions)
        mentions.extend(bert_mentions)

        for token in mentions:
            entity_mention[token] = token
        return entity_mention
    
    def restore_entities(self, pred, q):
        entities = []
        str = ''
        pred = pred[1:-1] # cls & sep
        for i in range(min(len(pred), len(q))):
            if pred[i]:
                str += q[i]
            else:
                if len(str):
                    entities.append(str)
                    str = ''
        if len(str):
            entities.append(str)
        return entities


if __name__ == '__main__':
    inputs = [
        '../data/corpus_train.json', 
        '../data/corpus_dev.json'
    ]
    outputs = [
        '../data/entity_mentions_train.json',
        '../data/entity_mentions_dev.json'
    ]
    s = "尿液肺炎链球菌抗原检测试验属于哪种类型？"
    me = MentionExtractor()
    me.extract_mentions(s)

    for in_path, out_path in zip(inputs, outputs):
        corpus = json.load(open(in_path, 'r', encoding='utf-8'))
        corpus = me.get_entity_mention(corpus)
        json.dump(
            corpus, 
            open(out_path, 'w', encoding='utf-8'), 
            indent=4, ensure_ascii=False
        )

