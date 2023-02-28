from TopicWordRecognization.run import predict as ner_predict
from CandidateTriplesSelection.run import predict as cls_predict
from CandidateTriplesLookup.knowledge_retrieval import entity_linking, search_triples_by_index
from AnswerRanking.ranking import get_attribution, similarity_attri_rela
import os
import datetime
import json
import re
from gensim.models import KeyedVectors

def load_word2vec():
    print('正在加载word2vec词向量', datetime.datetime.now())
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')
    print('word2vec词向量加载完毕', datetime.datetime.now())
    return word2vec_model

def pipeline_predict(question):
    ner_results = ner_predict(ner_checkpoint, question)
    ner_results = set([_result.replace('《', '').replace('》', '') for _result in ner_results])
    if not ner_results:
        ner_results = re.search(r'(.*)的.*是.*', question).group(1)
        if not ner_results:
            print('没有提取出主题词！')
            return
    print('■识别到的主题词：', ner_results, datetime.datetime.now())
    
    candidate_entities = []
    for mention in ner_results:
        candidate_entities.extend(entity_linking(mention2entity_dict, mention))
    print('■找到的候选实体：\n', candidate_entities, datetime.datetime.now())
    
    candidate_triples = search_triples_by_index(candidate_entities, forward_index, KB_file)
    candidate_triples = list(filter(lambda x: len(x) == 3, candidate_triples))
    cadidate_triples_num = len(candidate_triples)
    print('■候选三元组数量：', cadidate_triples_num, datetime.datetime.now())
    show_num = 20 if cadidate_triples_num > 20 else cadidate_triples_num
    print('■前{}条候选三元组：\n'.format(show_num), candidate_triples[:show_num])
    
    candidate_triples_labels = cls_predict(cls_checkpoint, [question] * len(candidate_triples), [triple[0] + triple[1] for triple in candidate_triples])
    predict_triples = [candidate_triples[i] for i in range(len(candidate_triples)) if candidate_triples_labels[i] == '1']
    print('■三元组粗分类结果，保留以下三元组：\n', predict_triples, datetime.datetime.now())
    
    predict_triples_num = len(predict_triples)
    if predict_triples_num == 0:
        print('■知识库中没有检索到相关知识，请换一个问题试试～')
        return
    elif predict_triples_num == 1:
        print('■预测答案唯一，直接输出')
        best_answer = predict_triples[0][2]
        print('■最佳答案：', best_answer)
    else:
        print('■预测答案多个，正在进行答案排序')
        max_ner_result = ''
        for _ner in ner_results:
            if len(_ner) > len(max_ner_result):
                max_ner_result = _ner
        attribution = get_attribution(question, max_ner_result)
        sim_cores = [similarity_attri_rela(word2vec_model, attribution, _triple[1]) for _triple in predict_triples]
        triples_with_score = list(zip(map(tuple, predict_triples), sim_cores))
        triples_with_score = sorted(triples_with_score, key=lambda x: x[1], reverse=True)
        print('■三元组排序结果：\n{}'.format('\n'.join([str(pair[0]) + '-->' + str(pair[1]) for pair in triples_with_score])))
        best_answer = triples_with_score[0][0][-1]
        print('■最佳答案：', best_answer)

if __name__ == '__main__':
    root_path = '/data/xiancai/cxc/BERT-KBQA-Pytorch'
    knowledge_path = '/data/xiancai/cxc/myRepo/datasets/nlpcc2016kbqa'
    knowledge_graph_path = os.path.join(knowledge_path, 'nlpcc-iccpol-2016.kbqa.kb')
    forward_index_path = os.path.join(root_path, 'CandidateTriplesLookup/data/forward_index.json')
    mention2entity_clean_path = os.path.join(root_path, 'CandidateTriplesLookup/data/mention2entity.json')
    
    print('正在加载mention2entity表', datetime.datetime.now())
    with open(mention2entity_clean_path, 'r', encoding='utf-8') as f:
        mention2entity_dict = json.load(f)
    print('mention2entity表加载完毕', datetime.datetime.now())
    
    print('正在加载知识库', datetime.datetime.now())
    KB_file = open(knowledge_graph_path, 'rb') # 二进制只读
    print('知识库加载完毕', datetime.datetime.now())

    print('正在加载索引表', datetime.datetime.now())
    with open(forward_index_path, 'r', encoding='utf-8') as f:
        forward_index = json.load(f)
    print('索引表加载完毕', datetime.datetime.now())

    word2vec_model_path = os.path.join(root_path, 'AnswerRanking/model/sgns.target.word-character')
    word2vec_model = load_word2vec()
    
    print('================================================================')
    
    ner_checkpoint = os.path.join(root_path, 'TopicWordRecognization/checkpoint')
    cls_checkpoint = os.path.join(root_path, 'CandidateTriplesSelection/checkpoint')
    
    while True:
        input_question = input('■请输入问题：')
        print('■已输入问句：', input_question)
        pipeline_predict(input_question)
        print('================================================================')