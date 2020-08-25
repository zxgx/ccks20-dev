import random
import numpy as np

from utils import cmp


def get_data(corpus):
    '''为验证集验证模型使用的数据
    X : numpy.array, (num_sample,num_feature)
    Y : numpy.array, (num_sample,1)
    samples : python-list,(num_sample,)
    ans : python-list, (num_question,num_answer)
    question2sample : python-dict, key:questionindex , value:sampleindexs
    '''
    X, Y, samples, ans = [], [], [], []
    gold_tuples, question2sample = [], {}
    
    sample_index, true_num, hop2_num, hop2_true_num = 0, 0, 0, 0
    for i in range(len(corpus)):
        candidate_tuples = corpus[i]['candidate_tuples']
        gold_entities = corpus[i]['gold_entities']
        gold_relations = corpus[i]['gold_relations']
        gold_tuple = tuple(gold_entities + gold_relations)
        answer = corpus[i]['answer']
        q_sample_indexs = []
        for t in candidate_tuples:
            features = candidate_tuples[t]
            if len(gold_tuple) == len(set(gold_tuple).intersection(set(t))):
                X.append([features[-1]])
                Y.append([1])
            else:
                X.append([features[-1]])
                Y.append([0])
            samples.append(t)
            q_sample_indexs.append(sample_index)
            sample_index+=1
        ans.append(answer)
        gold_tuples.append(gold_tuple)
        question2sample[i] = q_sample_indexs
        
        if_true = 0
        #判断gold tuple是否包含在候选tuples中
        for thistuple in candidate_tuples:
            if cmp(thistuple, gold_tuple)==0:
                if_true = 1
                break
        #判断单实体问题中，可召回的比例
        if if_true == 1:
            true_num += 1
            if len(gold_tuple) <=3 and len(gold_entities) == 1:
                hop2_true_num += 1
        if len(gold_tuple) <=3 and len(gold_entities) == 1:
            hop2_num += 1
        
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    print('单实体问题中，候选答案可召回的的比例为:%.3f'%(hop2_true_num/hop2_num))
    print('候选答案能覆盖标准查询路径的比例为:%.3f'%(true_num/len(corpus)))
    return X, Y, samples, ans, gold_tuples, question2sample


def get_train_data(corpus):
    '''
    为训练集的候选答案生成逻辑回归训练数据，由于正负例非常不均衡，对于负例进行0.05的采样
    '''
    X, Y = [], []
    true_num, hop2_num, hop2_true_num = 0, 0, 0

    for i in range(len(corpus)):
        candidate_tuples = corpus[i]['candidate_tuples']#字典
        gold_entities = corpus[i]['gold_entities']
        gold_relations = corpus[i]['gold_relations']
        gold_tuples = tuple(gold_entities + gold_relations)
        for t in candidate_tuples:
            features = candidate_tuples[t]
            if len(gold_tuples) == len(set(gold_tuples).intersection(set(t))):
                X.append([features[-1]]) # mention长度？？？？？？？？？？
                Y.append([1])
            else:
                prop = random.random()
                if prop<0.05:
                    X.append([features[-1]])
                    Y.append([0])
        
        if_true = 0 # 判断答案是否召回
        for thistuple in candidate_tuples:
            if cmp(thistuple, gold_tuples)==0:
                if_true = 1
                break
        if if_true == 1:
            true_num += 1
            if len(gold_tuples) <=3 and len(gold_entities) == 1:
                hop2_true_num += 1
        if len(gold_tuples) <=3 and len(gold_entities) == 1:
            hop2_num += 1
    
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    print('单实体问题中，候选答案可召回的的比例为:%.3f'%(hop2_true_num/hop2_num))
    print('候选答案能覆盖标准查询路径的比例为:%.3f'%(true_num/len(corpus)))
    return X, Y


def get_predict_tuples(prepro, samples, question2sample, topn):
    predict_tuples, predict_props = [], []

    for i in range(len(question2sample)):
        sample_indexs = question2sample[i]
        if len(sample_indexs) == 0:
            predict_tuples.append([])
            predict_props.append([])
            continue
        begin_index = sample_indexs[0]
        end_index = sample_indexs[-1]
        now_samples = [samples[j] for j in range(begin_index,end_index+1)]
        now_props = [prepro[j][1] for j in range(begin_index,end_index+1)]
        
        #(prop,(tuple))
        sample_prop = [each for each in zip(now_props,now_samples)]
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        tuples = [each[1] for each in sample_prop]
        props = [each[0] for each in sample_prop]
        predict_tuples.append(tuples[:topn])
        predict_props.append(props[:topn])
        
    return predict_tuples, predict_props


def compute_precision(gold_tuples, predict_tuples, predict_props):
    '''
    计算单实体问题中，筛选后候选答案的召回率，float
    '''
    true_num = 0
    one_subject_num = 0
    for i in range(len(gold_tuples)):
        gold_tuple = gold_tuples[i]
        if len(gold_tuple) <= 3:
            one_subject_num += 1
        for j in range(len(predict_tuples[i])):
            predict_tuple = predict_tuples[i][j]
            if cmp(predict_tuple, gold_tuple)==0:
                true_num += 1
                break
    return true_num/one_subject_num


def save_filter_candidate_tuple(corpus,predict_tuples):
    for i in range(len(corpus)):
        candidate_tuple_filter = {}
        for t in predict_tuples[i]:
            features = corpus[i]['candidate_tuples'][t]
            # print(features)
            new_features = features[0:2]+[features[-1]]
            # print(new_features)
            candidate_tuple_filter[t] = new_features
        corpus[i]['candidate_tuple_filter'] = candidate_tuple_filter
        #temp =corpus[i].pop('candidate_tuples')
    return corpus


if __name__ == '__main__':
    import pickle
    
    dev_path = '../data/candidate_tuples_dev.pkl'
    with open(dev_path, 'rb') as f:
        dev_corpus = pickle.load(f)
    train_path = '../data/candidate_tuples_train.pkl'
    with open(train_path, 'rb') as f:
        train_corpus = pickle.load(f)
    
    x_train, y_train = get_train_data(train_corpus)
    x_dev, y_dev, samples_dev, ans_dev, gold_tuples_dev,\
        question2sample_dev = get_data(dev_corpus)
    print(x_train.shape, y_train.shape)
    
    #逻辑回归
    from sklearn.preprocessing import StandardScaler
    from sklearn.externals import joblib
    from sklearn import linear_model
    import pickle
    
    sc = StandardScaler()
    sc.fit(x_train)
    joblib.dump(sc, '../data/model/tuple_scaler')
    x_train = sc.transform(x_train)
    x_dev = sc.transform(x_dev)
    
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(x_train, y_train)
    print(model.coef_)
    with open('../data/model/tuple_classifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    y_predict = model.predict_proba(x_dev).tolist()
    
    topns = [1,5,10,20,30]
    # topns = [10]
    for topn in topns:
        predict_tuples_dev, predict_props_dev = get_predict_tuples(
                y_predict, samples_dev, question2sample_dev, topn)
        precision_topn = compute_precision(
            gold_tuples_dev, predict_tuples_dev, predict_props_dev)
        print ('在验证集上逻辑回归筛选后top%d 召回率为%.2f'%(topn, precision_topn))
    
    dev_corpus = save_filter_candidate_tuple(dev_corpus, predict_tuples_dev)
    
    x_train, y_train, samples_train, ans_train, gold_tuples_train,\
        question2sample_train = get_data(train_corpus)
    y_predict = model.predict_proba(x_train).tolist()
    predict_tuples_train, predict_props_train = get_predict_tuples(
        y_predict, samples_train, question2sample_train,topn)
    train_corpus = save_filter_candidate_tuple(train_corpus, predict_tuples_train)
    
    train_path = '../data/candidate_tuples_filter_train.pkl'
    with open(train_path, 'wb') as f:
        pickle.dump(train_corpus, f)
    dev_path = '../data/candidate_tuples_filter_dev.pkl'
    with open(dev_path, 'wb') as f:
        pickle.dump(dev_corpus, f)

