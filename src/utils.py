import jieba


def compute_entity_features(question, entity, relations):
    p_tokens = []
    for p in relations:
        p_tokens.extend(jieba.lcut(p[1:-1])) # <>
    p_chars = [char for char in ''.join(p_tokens)]
    
    q_tokens = jieba.lcut(question)
    q_chars = [char for char in question]
    
    e_tokens = jieba.lcut(entity[1:-1])
    e_chars = [char for char in entity[1:-1]]
    
    qe_feature = features_from_two_sequences(q_tokens,e_tokens) + features_from_two_sequences(q_chars,e_chars)
    qr_feature = features_from_two_sequences(q_tokens,p_tokens) + features_from_two_sequences(q_chars,p_chars)
    #实体名和问题的overlap除以实体名长度的比例
    return qe_feature+qr_feature

def features_from_two_sequences(s1, s2):
    overlap = len(set(s1)&set(s2))
    jaccard = len(set(s1)&set(s2)) / len(set(s1)|set(s2))
    
    return [overlap, jaccard]


def cmp(t1, t2):
    '''比较两个tuple是否相等'''
    if len(t1) != len(t2):
        return 1
    t1 = set(t1)
    t2 = set(t2)
    if len(t1) == len(t1.intersection(t2)) and len(t2)==len(t1.intersection(t2)):
        return 0
    else:
        return 1


def 


if __name__ == '__main__':
    pass
