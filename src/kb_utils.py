import json

from GstoreConnector import GstoreConnector


gc = GstoreConnector('pkubase.gstore.cn', 80, 'endpoint', '123')

def get_relations_2hop(entity):
    sparql = "select distinct ?x ?y where {{{a} ?x ?b. ?b ?y ?z}}".format(a=entity)
    
    paths = []
    res = json.loads(gc.query('pkubase', 'json', sparql)
    try:
        for each in res['results']['bindings']:
            paths.append([each['x']['value'], each['y']['bindings']])
        ret = dict()
        for path in paths:
            for p in path:
                ret[p] = 0
    except:
        ret = dict()
    return ret
    
def get_relation_num(entity):
    sparql = "select ?x  where {{{a} ?x ?y}}".format(a=entity)
    res = json.loads(gc.query('pkubase', 'json', sparql))
    
    try:
        return len(res['results']['bindings'])
    except:
        return 0


if __name__ == '__main__':
    pass
