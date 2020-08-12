import json
import time

not_pos = {'f','d','h','k','r','c','p','u','y','e','o','g','w','m'}
pass_mention_dict = {
    '是什么', '在哪里', '哪里', '什么', '提出的', '有什么', '国家', '哪个', '所在', '培养出', '为什么', 
    '什么时候', '人', '你知道', '都包括', '是谁', '告诉我', '又叫做', '有', '是'
}

class SubjectExtractor():
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
            
    
    
