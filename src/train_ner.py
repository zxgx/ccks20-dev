from transformers import BertTokenizer, BertModel
import json


max_seq_len = 20
BERT_ID = 'hfl/chinese-bert-wwm-ext'


def main():
    tokenizer = BertTokenizer.from_pretrained(BERT_ID)
    
    # 训练集
    with open('../data/corpus_train.json', encoding='utf-8') as f:
        train_corpus = json.load(f)
    train_questions = [ train_corpus[i]['question'] \
        for i in range(len(train_corpus)) ]
    train_entities = [ train_corpus[i]['gold_entities'] \
        for i in range(len(train_corpus)) ]
    train_entities = [ [entity[1:-1].split('_')[0] for entity in line] \
        for line in train_entities ]
    
    # 测试集
    with open('../data/corpus_test.json', encoding='utf-8') as f:
        test_corpus = json.load(f)
    test_questions = [ test_corpus[i]['question'] \
        for i in range(len(test_corpus)) ]
    test_entities = [ test_corpus[i]['gold_entities'] \
        for i in range(len(test_corpus)) ]
    test_entities = [ [entity[1:-1].split('_')[0] for entity in line] \
        for line in test_entities ]
    

def find_lcsubstr(s1, s2): 
    m=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] #生成0矩阵，为方便后续计算，比字符串长度多了一列
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


def batch_data(questions, entities):
    x1, x2, y = [], [], []
    for i in range(len(questions)):
        q = question[i]
        