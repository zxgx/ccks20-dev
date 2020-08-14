import json
import time
import jieba.posseg as pseg

from kb_utils import get_relation_num, get_relations_2hop
from utils import compute_entity_features

# jieba的非paddle词性标记模型中，标点符号的词性为非语素词x
stop_pos = {'f','d','h','k','r','c','p','u','y','e','o','g','x','m'}

stop_mention = {
    '是什么', '在哪里', '哪里', '什么', '提出的', '有什么', '国家', '哪个', '所在', '培养出', '为什么', 
    '什么时候', '人', '你知道', '都包括', '是谁', '告诉我', '又叫做', '有', '是'
}

class EntityExtractor():
    def __init__(self):
        with open('../data/mention2entity.json', 'r', encoding='utf-8') as f:
            self.mention2entity = json.load(f)
        
        try:
            with open('../data/entity2hop.json', 'r', encoding='utf-8') as f:
                self.entity2hop = json.load(f)
        except:
            self.entity2hop = dict()
        
        self.word2freq = self.load_word2freq('../corpus/SogouLabDic.dic')
        
        self.f = open('../data/record/entity_extractor_ans.txt')
        
    def load_word2freq(self, path):
        ret = dict()
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]
        for line in lines:
            wft = line.split('\t')
            word = wft[0]
            f = int(wft[1]) // 10000
            ret[word] = f
        return ret
    
    def get_candidate_entity(self, corpus): 
        num_true, num_one, num_one_true, num_subject = 0., 0., 0., 0.
        for i in range(len(corpus)):
            item = corpus[i]
            question, gold_entities = item['question'], item['gold_entities']
            
            st = time.time()
            entity_mention = item['entity_mention']
            subject_properties = item['subject_properties']
            candidate_entities = self.extract_subject(
                entity_mention, subject_properties , question)
            num_subject += len(candidate_entities)
            item['candidate_entities'] = candidate_entities
            print(i, question)
            print('候选实体：')
            for c in candidate_entities:
                print(c, candidate_entities[c])
            print(len(candidate_entities))
            print('时间:%.2fs'%(time.time()-st))
            
            if len(set(gold_entities)) == len(set(gold_entities).intersection(\
            set(candidate_entities))): # 实体抽取成功
                num_true += 1
                if len(gold_entities) == 1:
                    num_one_true
            else: # 实体抽取失败
                self.f.write(str(i)+': '+question+'\n')
                self.f.write('\t'.join(gold_entities)+'\n')
                self.f.write('\t'.join(list(candidate_entities.keys()))+'\n\n')
            if len(gold_entities) == 1:
                num_one += 1
        
        print('单实体问题可召回比例为：%.2f'%(num_one_true/num_one))
        print('所有问题可召回比例为：%.2f'%(num_true/len(corpus)))
        print('平均每个问题的候选主语个数为：%.2f'%(num_subject/len(corpus)))
        
        with open('../data/entity2hop.json', 'w', encoding='utf-8') as f:
            json.dump(self.entity2hop, f, indent=4, ensure_ascii=False)
        
        return corpus

    def extract_subject(self, entitiy_mentions, property_mentions, question):
        candidates = dict()
        for mention in entitiy_mentions:
            pos = pseg.lcut(mention)
            if len(pos) == 1 and pos[0].flag in stop_pos:
                continue
            if mention in stop_mention:
                continue
            
            if mention in self.mention2entity:
                for ent in self.mention2entity[mention]:
                    # mention特征
                    mention_features = self.get_mention_features(
                        question, mention)
                    
                    # 实体两跳内所有关系
                    entity = '<' + ent + '>'
                    if entity in self.entity2hop:
                        relations = self.entity2hop[entity]
                    else:
                        relations = get_relations_2hop(entity)
                        self.entity2hop[entity] = relations
                    
                    # 问题和主语实体及其两跳内关系间的相似度
                    similar_features = compute_entity_features(
                        question, entity, relations)
                    
                    # 实体的流行度特征
                    popular_feature = get_relation_num(entity)
                    candidate[entity] = mention_features + similar_features +\
                        [popular_feature ** 0.5]
        
        for property in property_mentions:
            mention = property_mentions[property]
            if mention in self.stop_mention or property in self.stop_mention:
                continue
            pos = pseg.lcut(mention)
            if len(pos) == 1 and pos[0].flag in stop_pos:
                continue
            
            # 实体？属性值？这里或许不用过滤？
            entity = '<' + property + '>'
            if entity in candidate:
                continue
            
            # mention特征
            mention_features = self.get_mention_features(question, mention)
            
            # 实体两跳内所有关系
            entity = '"' + property + '"'
            if entity in self.entity2hop:
                relations = self.entity2hop[entity]
            else:
                relations = get_relations_2hop(entity)
                self.entity2hop[entity] = relations
            
            # 问题和主语实体及其两跳内关系间的相似度
            similar_features = compute_entity_features(
                question, entity, relations)
            popular_feature = get_relation_num(entity)
            candidate[entity] = mention_features + similar_features +\
                [popular_feature ** 0.5]

        return candidate
    
    def get_mention_features(self, question, mention):
        f1 = float(len(mention))
        
        try:
            f2 = float(self.word2freq[mention])
        except: # OOV mention
            f2 = 1.0
        
        if mention[-2:] == '大学':
            f2 = 1.0
        
        try:
            f3 = float(question.index(mention))
        except:
            f3 = 3.0
        
        return [mention, f1, f2, f3] # mention???


if __name__ == "__main__":
    inputs = [
        '../data/all_mentions_train.json',
        '../data/all_mentions_dev.json',
        '../data/all_mentions_test.json'
    ]
    outputs = [
        '../data/candidate_entities_train.json',
        '../data/candidate_entities_dev.json',
        '../data/candidate_entities_test.json'
    ]
    
    se = EntityExtractor()
    
    for in_path, out_path in zip(inputs, outputs):
        with open(in_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        corpus = se.get_candidate_entity(corpus)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=4, ensure_ascii=False)
    se.f.close()
