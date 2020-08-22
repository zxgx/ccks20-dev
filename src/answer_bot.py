import pickle
import re
import json
import numpy as np

import jieba.posseg as pseg
from collections import defaultdict

from mention_extractor import MentionExtractor
from entity_extractor import EntityExtractor
from property_extractor import PropertyExtractor
from tuple_extractor import TupleExtractor
from kb_utils import gc

# 双实体问题桥接不考虑的关系
not_relation = {'<中文名>', '<外文名>', '<本名>', '<别名>', '<国籍>', '<职业>'}

class KBQA():
    def __init__(self):
        
        self.mention_extractor = MentionExtractor()
        self.property_extractor = PropertyExtractor()
        self.entity_extractor = EntityExtractor()
        self.tuple_extractor = TupleExtractor()
        
        self.topn_e = 5
        self.topn_t = 3
        
        path = '../data/model/entity_classifier_model.pkl'
        with open(path, 'rb') as f:
            self.entity_classifier = pickle.load(f)
        
        path = '../data/model/tuple_classifier_model.pkl'
        with open(path, 'rb') as f:
            self.tuple_classifier = pickle.load(f)
    
    def add_answers_to_corpus(self, corpus):
        for sample in corpus:
            question = sample['question']
            ans = self.answer_main(question)
            sample['predict_ans'] = ans
        
        return corpus
    
    def answer_main(self, question):
        '''
        输入问题，依次执行：
        抽取实体mention、抽取属性值、生成候选实体并得到特征、候选实体过滤、
        生成候选查询路径（单实体双跳）、候选查询路径过滤
        使用top1的候选查询路径检索答案并返回
        input:
            question : python-str
        output:
            answer : python-list, [str]
        '''
        dic = {}
        print('>>> 原问题')
        print(question)
        
        question = re.sub('在金庸的小说《天龙八部》中，','',question)
        question = re.sub('电视剧武林外传里','',question)
        question = re.sub('《射雕英雄传》里','',question)
        question = re.sub('情深深雨濛濛中','',question)
        question = re.sub('《.+》中','',question)
        question = re.sub('常青藤大学联盟中','',question)
        question = re.sub('原名','中文名',question)
        question = re.sub('英文','外文',question)
        question = re.sub('英语','外文',question)
        
        dic['question'] = question
        print('>>> re处理后的问题')
        print(question)
        
        mentions = self.mention_extractor.extract_mentions(question)
        dic['mentions'] = mentions
        print('>>> 实体mention抽取结果')
        print(list(mentions.keys()))
        
        properties = self.property_extractor.extract_properties(question)
        subject_properties, special_properties = self.add_properties(
            mentions, properties)
        dic['properties'] = subject_properties
        print('>>> 属性mention抽取结果')
        print(list(subject_properties.keys()))
        
        subjects = self.entity_extractor.extract_subject(
            mentions, subject_properties, question)
        dic['subjects'] = subjects
        print('>>> 主语实体')
        print(list(subjects.keys()))
        if len(subjects) == 0:
            return []
        
        subjects = self.subject_filter(subjects)
        for properties in special_properties:
            sub = '"' + properties + '"'
            if sub not in subjects:
                subjects[sub] = [special_properties[properties], 3, 1, 1, 2, 6]
        dic['subjects_filter'] = subjects
        print('>>> 筛选后的主语实体')
        print(list(subjects.keys()))
        if len(subjects) == 0:
            return []
        
        tuples = self.tuple_extractor.extract_tuples(subjects, question)
        dic['tuples'] = tuples
        print('>>> 抽取的查询路径')
        print(tuples)
        if len(tuples) == 0:
            return []
        
        tuples = self.tuple_filter(tuples)
        dic['tuples_filter'] = tuples
        print('>>> 筛选后的查询路径')
        print(tuples)
        
        top_tuple = self.get_most_overlap_tuple(question, tuples)
        print('>>> 最终查询路径')
        print(top_tuple)
        
        # 生成查询语句
        search_paths = [ each for each in top_tuple ]
        if len(search_paths) == 2:
            sparql = 'select ?x where {{{a} {b} ?x}}'.format(\
                a=search_paths[0], b=search_paths[1])
            ret = json.loads(gc.query('pkubase', 'json', sparql))
            ans = [ each['x']['value'] for each in ret['results']['bindings'] ]
        elif len(search_paths) == 3:
            sparql = 'select ?x where {{{a} {b} ?m . ?m {c} ?x}}'.format(\
                a=search_paths[0], b=search_paths[1], c=search_paths[2])
            ret = json.loads(gc.query('pkubase', 'json', sparql))
            ans = [ each['x']['value'] for each in ret['results']['bindings'] ]
        elif len(search_paths) == 4:
            sparql = 'select ?x where {{{a} {b} ?x . ?x {c} {d}}}'.format(\
                a=search_paths[0], b=search_paths[1], 
                c=search_paths[2], d=search_paths[3])
            ret = json.loads(gc.query('pkubase', 'json', sparql))
            ans = [ each['x']['value'] for each in ret['results']['bindings'] ]
        else:
            print('不规范的查询路径')
            ans = []
        dic['answer'] = ans
        
        print('>>> 答案为')
        print(ans, '\n')
        
        return ans
    
    def add_properties(self, mentions, properties):
        '''
        mentions: {entity - mention} ???
        properties: {class - {entity - mention}}
        '''
        subject_properties = dict()
        subject_properties.update(properties['mark_properties'])
        subject_properties.update(properties['time_properties'])
        subject_properties.update(properties['digit_properties'])
        subject_properties.update(properties['other_properties'])
        subject_properties.update(properties['fuzzy_properties'])
        
        special_properties = dict()
        special_properties.update(properties['mark_properties'])
        special_properties.update(properties['time_properties'])
        
        return subject_properties, special_properties
    
    def subject_filter(self, subjects):
        '''
        输入候选主语和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选主语
        subjects: [entity - feature]
        '''
        entities, features = [], []
        for s in subjects:
            entities.append(s)
            features.append(subjects[s][1:])
        pred_prob = self.entity_classifier.predict_proba(
            np.array(features))[:, 1].tolist()
        sample_property = [each for each in zip(pred_prob, entities]
        sample_property = sorted(
            sample_property, key=lambda x:x[0], reverse=True)
        entities = [each[1] for each in sample_property]
        pred_entities = entities[:self.topn_e]
        
        new_subjects = dict()
        for e in pred_entities:
            new_subjects[e] = subjects[e]
        return new_subjects
    
    def tuple_filter(self, tuples):
        '''
        输入候选答案和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选答案
        tuples: tuple - feature
        '''
        tuple_list, features = [], []
        for t in tuples:
            tuple_list.append(t)
            features.append(tuples[t][-1:])
        xxx = features
        pred_prob = self.tuple_classifier.predict_proba(xxx)[:, 1].tolist()
        
        sample_prop = [each for each in zip(pred_prob, tuple_list)]
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        tuples_sorted = [each[1] for each in sample_prop]
        return tuples_sorted[:self.topn_t]        
    
    def get_most_overlap_tuple(self, question, tuples):
        '''
        从排名前几的tuples里选择与问题overlap最多的
        '''
        max_, ans = tuples[0]
        for t in tuples:
            text = ''
            for element in t:
                element = element[1:-1].split('_')[0]
                text += element
            f1 = len(set(text).intersection(set(question)))
            f2 = f1/len(set(text))
            f = f1 + f2
            if f > max_:
                max_, ans = f, t
        return ans


if __name__ == '__main__':
    import json
    
    qa = KBQA()
    
    path = '../data/candidate_entities_filter_test.json'
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    corpus = qa.add_answers_to_corpus(corpus)
    
    ave_f = 0.0
    for i in range(len(corpus)):
        sample = corpus[i]
        gold_ans = sample['answer']
        pred_ans = sample['predict_ans']
        # f1计算，感觉有点问题？？？
        true = len(set(gold_ans).intersection(set(pred_ans)))
        precision = true / len(set(pred_ans))
        recall = true / len(set(gold_ans))
        try:
            f1 = 2*precision*recall/(precision+recall)
        except:
            f1 = 0.
        ave_f += f1
    ave_f /= len(corpus)
    print(ave_f)
