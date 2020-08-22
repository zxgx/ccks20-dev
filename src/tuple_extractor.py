from similarity import BertSim, BERT_ID, predict
from transformers import BertTokenizer
from kb_utils import get_relation_paths
import torch

class TupleExtractor(object):
    def __init__(self):
        #加载微调过的文本匹配模型    
        self.simmer = BertSim()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        self.device = torch.device('cuda:1')
        print ('bert相似度匹配模型加载完成')
        
        #加载简单-复杂问题分类模型
        #self.question_classify_model = get_model()
        print ('问题分类模型加载完成')
        print ('tuple extractor loaded')
    
    def extract_tuples(self, candidate_entitys, question):
        ''''''
        candidate_tuples = {}
        entity_list = candidate_entitys.keys() # 得到有序的实体列表
        count, st = 0, time.time() # 获取所有候选路径的BERT输出
        for entity in entity_list:
            #得到该实体的所有关系路径
            relations = get_relation_paths(entity)
            mention = candidate_entitys[entity][0]
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

        print('====共有{}个候选路径===='.format(count)
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
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            gold_entities = dic['gold_entities']
            gold_relations = dic['gold_relations']
            gold_tuple = tuple(gold_entities + gold_relations)
            candidate_entitys = dic['candidate_entity_filter']
            print (i)
            print (question)
            candidate_tuples = self.extract_tuples(candidate_entitys, question)
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
                if len(gold_tuple) <=3 and len(gold_entitys) == 1:
                    hop2_true_num += 1
            if len(gold_tuple) <=3 and len(gold_entitys) == 1:
                hop2_num += 1
                
        print('所有问题里，候选答案能覆盖标准查询路径的比例为:%.3f'%(true_num/len(corpus)))
        print('单实体问题中，候选答案能覆盖标准查询路径的比例为:%.3f'%(hop2_true_num/hop2_num))
        print('平均每个问题的候选答案数量为:%.3f'%(all_tuples_num/len(corpus)))
        
        return corpus

if __name__ == '__main__':
    import json
    
    inputpaths = ['../data/candidate_entitys_filter_train.json',
                  '../data/candidate_entitys_filter_test.json',
                  '../data/candidate_entitys_filter_valid.json']
    outputpaths = ['../data/candidate_tuples_train.json',
                   '../data/candidate_tuples_test.json',
                   '../data/candidate_tuples_valid.json']
    te = TupleExtractor()
    for inputpath,outputpath in zip(inputpaths,outputpaths):
        with open(inputpath, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        corpus = te.get_candidate_ans(corpus)
        with open(outputpath, 'w', encoding='utf-8') as f:
            json.dum(corpus, f, indent=4, ensure_ascii=False)
