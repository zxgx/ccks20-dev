import json
import thulac


class PropertyExtractor():
    def __init__(self):
        self.property_dict = json.load(
            open('../data/property_dict.json', 'r', encoding='utf-8'))
        self.char2property = json.load(
            open('../data/char2property.json', 'r', encoding='utf-8'))
        self.segger = thulac.thulac()

    def get_proprerties(self, corpus):
        gold_num = 0


if __name__ == '__main__':
    pass
