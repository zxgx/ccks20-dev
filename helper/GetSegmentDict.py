import json


def get_segment_dict():
    segment_dict = dict()
    
    with open('../PKUBASE/pkubase-complete2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            entity = line.strip().split('\t')[0]
            entity_name = entity[1:-1]
            if '_' in entity_name:  # 多个实体有多个下划线
                entity_name = entity_name.split('_')[0]
            segment_dict[entity_name] = 1
    print('pkubase读取完成')

    with open('../PKUBASE/pkubase-mention2ent.txt', 'r', encoding='utf-8') as f:
        for line in f:
            entity_name = line.strip().split('\t')[0]
            segment_dict[entity_name] = 1
    print('mention2entity读取完成')
    print('分词词典大小为: %d'%len(segment_dict))

    with open('../data/segment_dict.txt', 'w', encoding='utf-8') as f:
        for e in segment_dict:
            f.write(e+'\n')


def get_property_dict():
    property_dict = dict()
    with open('../PKUBASE/pkubase-complete2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ob = line.split('\t')[2][:-3] # "literal[" .]
                if ob[0] == '"':
                    literal = ob[1:-1]
                    if literal in property_dict:
                        property_dict[literal] += 1
                    else:
                        property_dict[literal] = 1
            except: # 有几条三元组格式非 sbj\trel\tobj . 
                print('非标准格式三元组:', line.strip())
                continue
    
    print('pkubase属性值读取完成')
    print('属性值词典大小为: %d'%len(property_dict))

    with open('../data/property_dict.json', 'w', encoding='utf-8') as f:
        json.dump(property_dict, f, ensure_ascii=False, indent=4)


def main():
    get_segment_dict()

    get_property_dict()


if __name__ == '__main__':
    main()

