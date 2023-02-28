import jieba

def get_attribution(question, ner_result):
    '''
    抽取提问属性词
    '''
    question = question.replace(ner_result, '').replace('《', '').replace('》', '')
    for word in ['我想知道','我想请问','请问你','请问','你知道','谁知道','知道','谁清楚','我很好奇','你帮我问问','有没有人看过','有没有人'
                '怎么','这个','有多少个','有哪些','哪些','哪个','多少','几个','谁','被谁','还有'
                ,'吗','呀','啊','吧','着','的','是','呢','了','？','?','什么']:
        question = question.replace(word, '')
    return question

def similarity_attri_rela(word2vec_model, attribution_text, relation_text):
    return Jaccard_similarity(attribution_text, relation_text) + Word2vec_similarity(word2vec_model, attribution_text, relation_text)

def Jaccard_similarity(string1, string2):
    '''
    Jaccard 系数定义为A与B交集的大小与A与B并集的大小的比值
    '''
    intersection = len(set(string1).intersection(set(string2)))
    union = len(set(string1).union(set(string2)))
    return intersection / union

def Word2vec_similarity(word2vec_model, string1, string2):
    words1 = jieba.cut(string1)
    words2 = jieba.cut(string2)
    
    seg1, seg2 = [], []
    for word in words1:
        if word not in word2vec_model.vocab:
            ws = [w for w in word if w in word2vec_model.vocab]
            seg1.extend(ws)
        else:
            seg1.append(word)
    for word in words2:
        if word not in word2vec_model.vocab:
            ws = [w for w in word if w in word2vec_model.vocab]
            seg2.extend(ws)
        else:
            seg2.append(word)
    if seg1 and seg2:
        score = word2vec_model.n_similarity(seg1, seg2)
    else:
        score = 0
    return score