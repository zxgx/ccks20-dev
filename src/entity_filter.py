'''
该模块目的是从所有候选实体中筛选出topn个候选实体,并保证正确实体在其中
'''

import numpy as np


def get_data(corpus):
    X, Y, samples = [], [], []
    gold_entities, question2sample, sample_index= [], dict(), 0
    
    true_num, one_num, one_true_num = 0, 0, 0
    for i in range(len(corpus)):
        candidate_entity = corpus[i]['candidate_entities']
        gold_entity = corpus[i]['gold_entities']
        
        candidate_entitys_list = [each for each in candidate_entity]
        if len(gold_entity) == len(set(gold_entity).intersection(
            set(candidate_entitys_list))):
            true_num += 1
            if len(gold_entity) == 1:
                one_true_num += 1
        if len(gold_entity) == 1:
            one_num += 1

        q_sample_indexs = []
        for e in candidate_entity:
            features = candidate_entity[e]
            X.append(features[1:])  # 第0个特征是该实体对应的mention 
            if e in gold_entity:
                Y.append(1)
            else:
                Y.append(0)
            samples.append(e)
            q_sample_indexs.append(sample_index)
            sample_index+=1
        gold_entities.append(gold_entity)
        question2sample[i] = q_sample_indexs  # 每个问题i对应的sample index
    print ('所有问题候选主语召回率为：%.3f 其中单主语问题为：%.3f'%\
    (true_num/len(corpus),one_true_num/one_num))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    return X, Y, samples, gold_entities, question2sample


def get_predict_entities(predict_prob, samples, question2sample, topn):
    '''
    得到问题对应的样本，对它们按照概率进行排序，选取topn作为筛选后的候选实体
    对于属性值，只保留排名前3位的 ???
    '''
    predict_entities = []
    for i in range(len(question2sample)):
        sample_indexs = question2sample[i]
        if len(sample_indexs) == 0:
            predict_entities.append([])
            continue
        begin_index = sample_indexs[0]
        end_index = sample_indexs[-1]
        now_samples = [samples[j] for j in range(begin_index,end_index+1)]
        now_props = [predict_prob[j][1] for j in \
            range(begin_index, end_index+1)]
        
        #(prop,(tuple))
        sample_prop = [each for each in zip(now_props,now_samples)]
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        entities = [each[1] for each in sample_prop]
        predict_entities.append(entities[:topn])

    return predict_entities


def compute_precision(gold_entities, predict_entities):
    '''
    判断每个问题预测的实体和真实的实体是否完全一致，返回正确率
    '''
    true_num, one_num, one_true_num = 0, 0, 0
    wrong_list = []#所有筛选实体错误的问题的序号
    for i in range(len(gold_entities)):
        if len(set(gold_entities[i]).intersection(set(predict_entities[i])))\
            == len(gold_entities[i]):
            true_num +=1
            if len(gold_entities[i]) == 1:
                one_true_num +=1
        else:
            if len(gold_entities[i]) == 1: # 只要有一个在gold中就不算错误
                wrong_list.append(i)
        if len(gold_entities[i])==1:
            one_num+=1
            
    return one_true_num/one_num, true_num/len(gold_entities), wrong_list


def save_filter_candidateE(corpus, predict_entities):
    for i in range(len(corpus)):
        candidate_entity_filter = dict()
        for e in predict_entities[i]:
            # print (corpus[i]['candidate_entities'][e])
            candidate_entity_filter[e] = corpus[i]['candidate_entities'][e]
        corpus[i]['candidate_entity_filter'] = candidate_entity_filter
    return corpus


if __name__ == '__main__':
    import json
    import pickle
    from sklearn import linear_model
    
    train_path = '../data/candidate_entities_train.json'
    with open(train_path, 'r', encoding='utf-8') as f:
        train_corpus = json.load(f)
    
    dev_path = '../data/candidate_entities_dev.json'
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_corpus = json.load(f)
    
    x_train, y_train, samples_train, gold_entities_train, question2sample_train = get_data(train_corpus)
    x_dev, y_dev, samples_dev, gold_entities_dev, question2sample_dev = get_data(dev_corpus)
    print(x_train.shape)
    
    # 逻辑回归
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(x_train, y_train)
    with open('../data/model/entity_classifer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    y_predict = model.predict_proba(x_dev).tolist() # (num_sample, 2-class)
    
    topns = [1,2,3,5,6,7,8,9,10,15,20]
    #得到候选实体
    for topn in topns:
        predict_entities= get_predict_entities(
            y_predict, samples_dev, question2sample_dev, topn)
        #判断候选实体的准确性，只要有一个在真正实体中即可
        precision_topn_one, precision_topn_all, wrong_list= compute_precision(
            gold_entities_dev, predict_entities)
        print ('在验证集上逻辑回归top%d筛选后，所有问题实体召回率为%.3f，单实体问题实体召回率%.3f'%(topn,precision_topn_all, precision_topn_one))
    #将筛选后的候选实体写入corpus并保存
    dev_corpus = save_filter_candidateE(dev_corpus, predict_entities)
    
    y_predict = model.predict_proba(x_train).tolist()
    predict_entities = get_predict_entities(
        y_predict, samples_train, question2sample_train, topn)
    train_corpus = save_filter_candidateE(train_corpus,predict_entities)
    
    train_path = '../data/candidate_entitys_filter_train.json'
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_corpus, f, indent=4, ensure_ascii=False)
    
    dev_path = '../data/candidate_entitys_filter_dev.json'
    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_corpus, f, indent=4, ensure_ascii=False)
