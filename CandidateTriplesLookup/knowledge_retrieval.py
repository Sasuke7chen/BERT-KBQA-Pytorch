import unicodedata
import re
import Levenshtein
import json

def entity_linking(mention2entity_dict, input_mention):
    if input_mention == '': return []
    input_mention = input_mention.replace(' ', '')
    relative_entities = mention2entity_dict.get(input_mention, [])
    if not relative_entities:
        fuzzy_query_relative_entities = dict()
        input_mention = unify_char_format(input_mention)
        for mention in mention2entity_dict.keys():
            prim_mention = mention # 初始实体
            mention = unify_char_format(mention)
            if len(mention) == 0:
                continue
            if '\\' == mention[-1]:
                mention = mention[:-1] + '\"'
            
            if ',' in mention or '、' in mention or ';' in mention or \
            '\\\\' in mention or ('或' in mention and '或' not in input_mention):
                mention_splits = re.split(r'[,;、或]|\\\\', mention)
                for _mention in mention_splits:
                    mention_len = len(input_mention)
                    editDistance = Levenshtein.distance(input_mention, _mention)
                    if (mention_len < 6 and editDistance <= 1) or (mention_len >= 6 and editDistance <= 4) or (mention_len >= 20 and editDistance <= 10):
                        fuzzy_query_relative_entities[prim_mention] = editDistance
            else:
                mention_len = len(input_mention)
                editDistance = Levenshtein.distance(input_mention, mention)
                if (mention_len < 6 and editDistance <= 1) or (mention_len >= 6 and editDistance <= 4) or (mention_len >= 20 and editDistance <= 10):
                    fuzzy_query_relative_entities[prim_mention] = editDistance
        if fuzzy_query_relative_entities:
            min_key = min(fuzzy_query_relative_entities.keys(), key=fuzzy_query_relative_entities.get)
            min_similar_score = fuzzy_query_relative_entities[min_key]
            for mention in fuzzy_query_relative_entities.keys():
                if fuzzy_query_relative_entities[mention] == min_similar_score:
                    relative_entities.extend(mention2entity_dict[mention])
        else:
            print('Oops! Find nothing but input')
    if input_mention not in relative_entities:
        relative_entities.append(input_mention)
    return relative_entities
                    

def search_triples_by_index(relative_entities, forward_index, raw_graph_f):
    relative_triples = []
    for entity in relative_entities:
        index_entity = forward_index.get(entity, None)
        if index_entity:
            read_index, read_size = index_entity['start_pos'], index_entity['length']
            raw_graph_f.seek(read_index)
            result = raw_graph_f.read(read_size).decode('utf-8')
            for line in result.strip().split('\n'):
                triple = line.strip().split(' ||| ')
                relative_triples.append(triple)
    return relative_triples

def unify_char_format(string):
    string = unicodedata.normalize('NFKC', string)
    string = string.replace('【', '[').replace('】', ']')
    string = string.lower()
    return string

if __name__ == '__main__':
    mention2entity_clean_path = './data/mention2entity.json'
    with open(mention2entity_clean_path, 'r', encoding='utf-8') as f:
        mention2entity_dict = json.load(f)
    input_mention = '史蒂芬霍金'
    rela_ents = entity_linking(mention2entity_dict, input_mention)
    print('匹配到知识库中的候选实体：', rela_ents)