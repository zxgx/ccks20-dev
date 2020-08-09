import json
from collections import defaultdict


max_seq_len = 20


if __name__ == '__main__':
    property_dict = json.load(
        open('../data/property_dict.json', 'r', encoding='utf-8'))

    char2property = defaultdict(list)

    for p in property_dict:
        if len(p) < max_seq_len:
            chars = set(p)
            for ch in chars:
                char2property[ch].append(p)

    json.dump(
        char2property, 
        open('../data/char2property.json', 'w', encoding='utf-8'),
        ensure_ascii=False,
        indent=4
    )

