import json
from GstoreConnector import GstoreConnector


DB_HOST = "pkubase.gstore.cn"
DB_NAME = "pkubase"
PORT = 80
USER_NAME = "endpoint"
PASSWORD = "123"


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            q = lines[i].split(':')
            '''
            if len(q) != 2:
                print(i, q)
                continue
            '''
            q_id = q[0]
            q_text = ':'.join(q[1:]).strip() # 或许会有问题
            sparql = lines[i+1].strip()
            ans = lines[i+2].strip().split('\t')
            data.append({
                'id':q_id, 'question': q_text, 'sparql': sparql, 'ans': ans
            })
    return data
    

if __name__ == '__main__':
    dataset_path = '../corpus/task1-4_train_2020.txt'
    dataset = load_data(dataset_path)
    
    idx = [357, 6288//4, 9844//4, 10084//4] # 后3个问题里含有‘:’
    for i in idx:
        print(dataset[i]['question'])
    
    print('\n'+'*'*50+'\n')
    
    gcon = GstoreConnector(DB_HOST, PORT, USER_NAME, PASSWORD)

    for i in range(len(dataset)):
        item = dataset[i]
        ret = json.loads(gcon.query(DB_NAME, 'json', item['sparql']))
        print(ret, '\n')
        if i == 15:
            break
    print('\n'+'*'*50+'\n')
    '''   
    max_seq_len = 0
    for i, item in enumerate(dataset):
        
        # 查询是一个json字符串
        # https://www.w3.org/TR/sparql11-overview/#sparql11-results
        ret = json.loads(gcon.query(DB_NAME, 'json', item['sparql'])) 
        
        if len(ret['head']['vars']) != 1: # 数据中含有多个变量的问题
            print(str(i)+'\n', item, '\n')
            continue

        var, values = ret['head']['vars'][0], []
        for v in ret['results']['bindings']:
            if v[var]['type'] == 'uri':
                values.append('<'+v[var]['value']+'>')
            else:
                values.append('"'+v[var]['value']+'"')
        for v in values:
            if v not in item['ans']:
                print(i)
                print(values)
                print(item['ans'], '\n')
                break
        if len(item['question']) > max_seq_len:
            max_seq_len = len(item['question'])
    print(max_seq_len)
    '''

