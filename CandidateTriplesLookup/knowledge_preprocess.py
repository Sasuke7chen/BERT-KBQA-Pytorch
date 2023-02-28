import json
import datetime
import os

def make_KG_index(knowledge_graph_path, forward_index_path):
    def make_index(graph_path, index_path):
        print('Begin to read KG', datetime.datetime.now())
        index_dict = dict()
        with open(graph_path, 'r', encoding='utf-8') as f:
            previous_entity = ''
            previous_start = 0
            while True:
                start_pos = f.tell()
                line = f.readline()
                if not line: break
                entity = line.split('|||')[0].strip()
                if previous_entity and entity != previous_entity:
                    tmp_dict = dict()
                    tmp_dict['start_pos'] = previous_start
                    tmp_dict['length'] = start_pos - previous_start
                    index_dict[previous_entity] = tmp_dict
                    previous_start = start_pos
                previous_entity = entity
        print('Finish reading KG, begin to write', datetime.datetime.now())
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_dict, f, indent=2, ensure_ascii=False)
    make_index(knowledge_graph_path, forward_index_path)
    
def clean_mention2entity(mention2entity_path, mention2entity_clean_path):
    mention2entity_clean = dict()
    # 7623035
    with open(mention2entity_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if '|||' in line:
                mention = line.split('|||')[0].strip()
                entity = line.split('|||')[1].strip()
                mention = mention.replace(' ', '')
                entity = entity.split('\t')
                mention2entity_clean[mention] = entity
    with open(mention2entity_clean_path, 'w', encoding='utf-8') as f:
        json.dump(mention2entity_clean, f, indent=2, ensure_ascii=False)
        
if __name__ == '__main__':
    knowledge_path = '/data/xiancai/cxc/myRepo/datasets/nlpcc2016kbqa'
    knowledge_graph_path = os.path.join(knowledge_path, 'nlpcc-iccpol-2016.kbqa.kb')
    forward_index_path = './data/forward_index.json'
    mention2entity_path = os.path.join(knowledge_path, 'nlpcc-iccpol-2016.kbqa.kb.mention2id')
    mention2entity_clean_path = './data/mention2entity.json'
    
    # make_KG_index(knowledge_graph_path, forward_index_path)
    # clean_mention2entity(mention2entity_path, mention2entity_clean_path)
    
    # with open(mention2entity_clean_path, 'r', encoding='utf-8') as f:
    #     mentions = json.load(f)
    #     for key, value in list(mentions.items())[90:100]:
    #         print(key, ' --> ', value) 