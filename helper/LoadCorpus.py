import re


def load_corpus(path):
    corpus = []
    question_num = 0
    e1hop1_num, e1hop2_num, e2hop2_num = 0, 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            # 问题
            line = lines[i].strip().split(':')
            idx = line[0]
            question = ':'.join(line[1:]) # 问题中含有多个冒号
            question = re.sub('你了解', '', question)
            question = re.sub('我想知道', '', question)
            question = re.sub('请问', '', question)
            
            # 答案
            answer = lines[i+2].strip().split('\t')
            
            # 查询，包含生成答案的正确路径
            sparql = lines[i+1].strip()
            where = re.findall('{.+}', sparql)[0]
            elements = re.findall('<.+?>|\".+?\"|\?\D', where) # 没有再加双引号项
            
            gold_entities, gold_relations = [], []
            for j in range(len(elements)):
                if elements[j][0] == '<' or elements[j][0] == '"':
                    if j%3 == 1:
                        gold_relations.append(elements[j])
                    else:
                        gold_entities.append(elements[j])
            gold_tuple = tuple(gold_entities+gold_relations)
            corpus.append({
                'id': idx,
                'question': question,
                'answer': answer,
                'gold_entities': gold_entities,
                'gold_relations': gold_relations,
                'sparql': sparql
            })
            
            if len(gold_entities) == 1 and len(gold_relations) == 1:
                e1hop1_num += 1
            elif len(gold_entities) == 1 and len(gold_relations) == 2:
                e1hop2_num += 1
            elif len(gold_entities) == 2 and len(gold_relations) == 2:
                e2hop2_num += 1
            elif len(gold_entities) == 2 and len(gold_relations) < 2:
                print(idx)
                print (elements)
                print (gold_entities)
                print (sparql)
                print ('\n')
            question_num += 1
    
    print("问题数为%d，单实体单关系为%d，单实体双关系为%d，双实体双关系为%d，总比例为%.4f\n"%
        (question_num, e1hop1_num, e1hop2_num, e2hop2_num, 
        (e1hop1_num+e1hop2_num+e2hop2_num)/question_num))
    return corpus


if __name__ == '__main__':
    import json
    import random

    corpus_path = '../corpus/task1-4_train_2020.txt'
    data_path = [
        '../data/corpus_train.json', 
        '../data/corpus_dev.json', 
        '../data/corpus_test.json'
    ]
    corpus = load_corpus(corpus_path)
    
    random.seed(1020)
    random.shuffle(corpus)
    train, dev, test = corpus[:3200], corpus[3200:3600], corpus[3600:]
    
    for data, path in zip([train, dev, test], data_path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

