from similarity import BertSim, BERT_ID, predict
from transformers import BertTokenizer
from kb_utils import get_relation_paths
import torch
import time


class TupleExtractor(object):
    def __init__(self):
        #加载微调过的文本匹配模型    
        self.simmer = BertSim()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        self.device = torch.device('cuda:0')
        self.simmer.load_state_dict(
            torch.load('../data/model/similarity.pt'))
        self.simmer.to(self.device)
        print ('bert相似度匹配模型加载完成')
        
        print ('tuple extractor loaded')
    
    def extract_tuples(self, candidate_entitys, question, entity2relations):
        ''''''
        candidate_tuples = {}
        entity_list = candidate_entitys.keys() # 得到有序的实体列表
        count, st = 0, time.time() 

        for entity in entity_list:
            mention = candidate_entitys[entity][0]
            relations = entity2relations[entity]
            for r in relations:
                #python-list 关系名列表
                predicates = [relation[1:-1] for relation in r]
                human_question = '的'.join([mention]+predicates)
                logits = predict(
                    self.simmer, self.tokenizer, self.device, 
                    question, human_question
                )
                sim = logits[0][1].item()
                
                this_tuple = tuple([entity] + r) # e, [r|r1, r2]
                # [entity, mention, feats]
                feature = [entity] + candidate_entitys[entity] + [sim]
                candidate_tuples[this_tuple] = feature
                count += 1

        print('====共有{}个候选路径===='.format(count))
        print('====为所有路径计算特征耗费%.2f秒===='%(time.time()-st))

        return candidate_tuples
    
    def get_candidate_ans(self, corpus):
        '''根据mention，得到所有候选实体,进一步去知识库检索候选答案
        候选答案格式为tuple(entity,relation1,relation2) 这样便于和标准答案对比
        '''
        true_num = 0
        hop2_num = 0
        hop2_true_num = 0
        all_tuples_num = 0
        
        relation_list, st = [], time.time()
        for i, item in enumerate(corpus):
            print(i)
            candidate_entity = item['candidate_entity_filter']
            entity_relation = dict()
            for e in candidate_entity:
                ret = get_relation_paths(e)
                entity_relation[e] = ret
                print('实体: %s查找到%d候选路径'%(e, len(ret)))
            relation_list.append(entity_relation)
            print()
        print('查询时间开销：%.2fs'%(time.time()-st))

        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            gold_entities = dic['gold_entities']
            gold_relations = dic['gold_relations']
            gold_tuple = tuple(gold_entities + gold_relations)
            candidate_entitys = dic['candidate_entity_filter']
            relations = relation_list[i]
            print (i)
            print (question)
            candidate_tuples = self.extract_tuples(
                candidate_entitys, question, relations)
            all_tuples_num += len(candidate_tuples)
            dic['candidate_tuples'] = candidate_tuples
            corpus[i] = dic
            
            #判断gold tuple是否包含在candidate_tuples_list中
            if_true = 0
            for thistuple in candidate_tuples:
                if len(gold_tuple) == len(set(gold_tuple)&set(thistuple)):
                    if_true = 1
                    break
            if if_true == 1:
                true_num += 1
                if len(gold_tuple) <=3 and len(gold_entities) == 1:
                    hop2_true_num += 1
            if len(gold_tuple) <=3 and len(gold_entities) == 1:
                hop2_num += 1
                
        print('所有问题里，候选答案能覆盖标准查询路径的比例为:%.3f'%(true_num/len(corpus)))
        print('单实体问题中，候选答案能覆盖标准查询路径的比例为:%.3f'%(hop2_true_num/hop2_num))
        print('平均每个问题的候选答案数量为:%.3f'%(all_tuples_num/len(corpus)))
        
        return corpus

if __name__ == '__main__':
    import json
    import pickle
    
    inputpaths = ['../data/candidate_entities_filter_train.json',
                  # '../data/candidate_entitys_filter_test.json',
                  '../data/candidate_entities_filter_dev.json']
    outputpaths = ['../data/candidate_tuples_train.pkl',
                   # '../data/candidate_tuples_test.pkl',
                   '../data/candidate_tuples_dev.pkl']
    te = TupleExtractor()
    for inputpath,outputpath in zip(inputpaths,outputpaths):
        with open(inputpath, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        corpus = te.get_candidate_ans(corpus)
        with open(outputpath, 'wb') as f:
            pickle.dump(corpus, f)

