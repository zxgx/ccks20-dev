import json

from GstoreConnector import GstoreConnector


gc = GstoreConnector('pkubase.gstore.cn', 80, 'endpoint', '123')

def get_relations_2hop(entity):
    sparql = "select distinct ?x ?y where {{{a} ?x ?b. ?b ?y ?z}}".format(a=entity)
    
    paths = []
    res = json.loads(gc.query('pkubase', 'json', sparql))
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


def get_relation_paths(entity):
    '''根据实体名，得到所有2跳内的关系路径，用于问题和关系路径的匹配'''
    sparql_1 = "select distinct ?x where {{{a} ?x ?y}}".format(a=entity)
    sparql_2 = "select distinct ?x ?y where {{{a} ?x ?b. ?b ?y ?z}}".format(a=entity)

    rpaths1, rpaths2 = [], []
    res1 = json.loads(gc.query('pkubase', 'json', sparql_1))
    res2 = json.loads(gc.query('pkubase', 'json', sparql_2))
    try:
        for each in res1["results"]["bindings"]:
            rpaths1.append([each["x"]["value"]])
    except:
        pass
    
    try:
        for each in res2["results"]["bindings"]:
            rpaths2.append([each["x"]["value"], each["y"]["value"]])
    except:
        pass
    
    return rpaths1 + rpaths2


def get_relation_paths_single(entity):
    pass


if __name__ == '__main__':
    sparql = "select ?x  where {{{a} ?x ?y}}".format(a='<周杰伦>')
    res = json.loads(gc.query('pkubase', 'json', sparql))
    print(res['results']['bindings'])
