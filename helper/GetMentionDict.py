from collections import defaultdict
import json


if __name__ == '__main__':
    mention2ent = defaultdict(list)
    with open('../PKUBASE/pkubase-mention2ent.txt', 'r', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')[:-1]
    for line in lines:
        link = line.split('\t')
        mention, entity = link[0], link[1]
        mention2ent[mention].append(entity)
    json.dump(
        mention2ent,
        open('../data/mention2entity.json', 'w', encoding='utf-8'),
        indent=4, ensure_ascii=False
    )
    
