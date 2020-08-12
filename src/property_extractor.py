import json
import re
import jieba.posseg as pseg


stop_dict = {
    '有', '的', '是', '在', '上', '哪', '里', '\"', '什', '么', '中', '个'
}

class PropertyExtractor():
    def __init__(self):
        self.property_dict = json.load(
            open('../data/property_dict.json', 'r', encoding='utf-8'))
        self.char2property = json.load(
            open('../data/char2property.json', 'r', encoding='utf-8'))

    def get_proprerties(self, corpus):
        gold_num, true_num = 0, 0
        entity_error, num_properties = [], 0.
        
        for i in range(len(corpus)):
            # gold properties
            gold_entities = corpus[i]['gold_entities']
            gold_properties = []
            for e in gold_entities:
                if e[0] == '"':
                    gold_properties.append(e)
            
            # 抽取的属性，不含引号
            pred_properties = self.extract_properties(corpus[i]['question'])
            corpus[i]['all_properties'] = pred_properties
            
            subject_properties = dict()
            subject_properties.update(pred_properties['mark_properties'])
            subject_properties.update(pred_properties['time_properties'])
            subject_properties.update(pred_properties['digit_properties'])
            subject_properties.update(pred_properties['other_properties'])
            subject_properties.update(pred_properties['fuzzy_properties'])
            
            corpus[i]['subject_properties'] = subject_properties
            num_properties += len(subject_properties)
            
            # 统计该模块抽取唯一主语实体的召回率
            if len(gold_properties) == 1 and len(gold_entities) == 1:
                gold_num += 1
                if_same = self.check_same(gold_properties, subject_properties)
                true_num += if_same
                if not if_same:
                    print('主语属性值抽取失败')
                    entity_error.append(i)
                else:
                    print('主语属性值抽取成功')
                print(i, corpus[i]['question'])
                print(gold_properties)
                print(subject_properties)
                print()
        print('单主语且主语为属性值问题中，能找到所有主语属性值的比例为:%.2f'%(true_num/gold_num))
        print('平均每个问题属性为:%.2f' % (num_properties / len(corpus)))
        print(entity_error)
        return corpus
        
    def extract_properties(self, q):
        '''
        从问题中抽取和知识库属性值匹配的字符串
        '''
        properties = dict()
        backup = q
        
        # 双引号和书名号
        mark_properties = dict()
        for e in re.findall('\".+\"|《.+》', q):
            if e in self.property_dict:
                mark_properties[e] = e
            q = re.sub(e, '', q)
        properties['mark_properties'] = mark_properties
        
        # 时间     issue：中文年月日、9.8、08.8
        time_properties = dict()
        # yyyy-mm-dd
        year_month_day = re.findall('\d+年\d+月\d+日|\d+年\d+月\d+号|\d+\.\d+\.\d+', q)
        for ymd in year_month_day:
            norm = self.normalize_date(ymd)
            time_properties[norm] = ymd
            q = re.sub(ymd, '', q)
        
        # mm-dd | yyyy-mm
        month_day = re.findall('\d+月\d+日|\d+月\d+号|\d+年\d+月', q)
        for ymd in month_day:
            norm = self.normalize_date(ymd)
            time_properties[norm] = ymd
            q = re.sub(ymd, '', q)
        
        # yyyy
        years = re.findall('\d+年', q)
        for ymd in years:
            norm = self.normalize_date(ymd)
            time_properties[norm] = ymd
            q = re.sub(ymd, '', q)
        properties['time_properties'] = time_properties
        
        # 数字
        digit_properties = dict()
        for e in re.findall('\d+', q):
            if e in self.property_dict:
                digit_properties[e] = e
            # q = re.sub(e, '', q) # 源码没有去掉数字?
        properties['digit_properties'] = digit_properties
        
        # 其他
        other_properties = dict()
        length, max_len = len(q), 0
        prop_ngram = []
        for l in range(length, 0, -1):
            # 长度至少为1，在属性值字典中
            for i in range(length+1-l):
                if q[i:i+l] in self.property_dict:
                    prop_ngram.append(q[i:i+l])
                    if len(q[i:i+l]) > max_len:
                        max_len = len(q[i:i+l])
        
        stop_props = []
        for pp in prop_ngram:
            for qq in prop_ngram:
                if pp != qq and pp in qq and pseg.lcut(pp)[0].flag != 'ns':
                    # 较短的，不是地名的属性值
                    stop_props.append(pp)
        
        new_props = [] # 去掉包含在更长属性值中的属性值
        for pp in prop_ngram:
            if pp not in stop_props:
                new_props.append(pp)
        
        new_new_props = [] # 去掉长度过于短的属性值
        for pp in new_props:
            if len(pp) == 1 and pseg.lcut(pp)[0].flag == 'n': # 单字名词
                new_new_props.append(pp)
            elif (len(pp) >= max_len*0.5) and len(pp) != 1 or pseg.lcut(pp)[0].flag in ['n', 'ns'] or self.exist_digit(pp):
                new_new_props.append(pp)
        
        for pp in new_new_props:
            other_properties[pp] = pp
        properties['other_properties'] = other_properties
        
        # 模糊匹配
        prop2num = dict()
        for ch in backup:
            if ch in stop_dict:
                continue
            else:
                try:
                    for pp in self.char2property[ch]:
                        if pp in prop2num:
                            prop2num[pp] += 1
                        else:
                            prop2num[pp] = 1
                except:
                    continue
        
        sort_props = sorted(prop2num.items(), key=lambda x:x[1], reverse=True)
        top3_props = [key for key,value in sort_props[:3]]
        fuzzy_properties = dict()
        for pp in top3_props:
            fuzzy_properties[pp] = pp
        properties['fuzzy_properties'] = fuzzy_properties
        
        return properties
    
    def normalize_date(self, date):
        elems = []
        for d in re.findall('\d+', date):
            if len(d) > 2: # 3位以上的数字，应该是年？
                elems.append(d)
            elif len(d) == 2:
                if int(d[0]) > 3: # 20年？
                    elems.append('19'+d)
                else: # 1929年这些？
                    elems.append(d)
            else:
                elems.append('0'+d)
        return '-'.join(elems)
    
    def exist_digit(self, p):
        for i in range(10):
            if str(i) in p:
                return True
        return False
    
    def check_same(self, gold, pred):
        pred_props_list = []
        for p in pred:
            pred_props_list.append('"'+p+'"')
        gold_props = set(gold)
        join_props = set(pred_props_list).intersection(gold_props)
        if len(join_props) == len(gold_props):
            return 1
        else:
            return 0


if __name__ == '__main__':
    inputs = [
        '../data/entity_mentions_train.json',
        '../data/entity_mentions_dev.json',
        '../data/entity_mentions_test.json'
    ]
    outputs = [
        '../data/all_mentions_train.json',
        '../data/all_mentions_dev.json',
        '../data/all_mentions_test.json'
    ]
    import time
    st = time.time()
    pe = PropertyExtractor()
    
    for in_path, out_path in zip(inputs, outputs):
        corpus = json.load(open(in_path, 'r', encoding='utf-8'))
        corpus = pe.get_proprerties(corpus)
        print('%s获取属性值实体'%in_path)
        json.dump(
            corpus,
            open(out_path, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )
    print('耗费时间%.2fs'%(time.time()-st))

