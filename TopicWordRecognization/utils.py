import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import Dataset
    
class NerDataset(Dataset):
    def __init__(self, data_path, label2id, max_seq_len, tokenizer):
        self.examples = self._read_text(data_path)
        self.label2id = label2id
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        return self.convert_example_to_feature(
            self.examples[index], 
            self.label2id, 
            self.max_seq_len, 
            self.tokenizer
        )
    
    def __len__(self):
        return len(self.examples)
    
    @staticmethod
    def _read_text(data_path):
        all_samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            tmp_sample_words, tmp_sample_labels = [], []
            for line in f.readlines():
                if line.strip() == '' and tmp_sample_words and tmp_sample_labels:
                    all_samples.append({"words": tmp_sample_words, "labels": tmp_sample_labels})
                    tmp_sample_words, tmp_sample_labels = [], []
                else:
                    word, label = line.strip().split(' ')[0], line.strip().split(' ')[1]
                    tmp_sample_words.append(word)
                    tmp_sample_labels.append(label)
            if tmp_sample_words:
                all_samples.append({"words": tmp_sample_words, "labels": tmp_sample_labels})
        return all_samples
    
    @staticmethod
    def convert_example_to_feature(example, label2id, max_seq_len, tokenizer):
        features = tokenizer.encode_plus(
            example["words"],
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
        )
        label_ids = [label2id[label] for label in example["labels"][: max_seq_len - 2]]
        label_ids = [label2id['O']] + label_ids + [label2id['O']]
        input_len = len(label_ids)
        padding_length = max_seq_len - input_len
        label_ids += [label2id['O']] * padding_length
        assert len(features["input_ids"]) == len(label_ids)
        input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(features["token_type_ids"], dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        return input_ids, token_type_ids, label_ids
    
class NerMetrics(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()
        
    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []
        
    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else 2 * recall * precision / (recall + precision)
        return recall, precision, f1
    
    def result(self):
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'Precision': precision, 'Recall': recall, 'F1': f1}
    
    def update(self, true_labels, pred_labels):
        for true_label, pred_label in zip(true_labels, pred_labels):
            true_entities = get_entities(true_label, self.id2label)
            pred_entities = get_entities(pred_label, self.id2label)
            self.origins.extend(true_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([pred_entity for pred_entity in pred_entities if pred_entity in true_entities])
    
def get_entities(pred_labels, id2label):
    # 单类实体
    chunks = []
    chunk = [-1, -1]
    for idx, label in enumerate(pred_labels):
        if not isinstance(label, str):
            label = id2label[label]
        if label == 'E':
            if chunk[0] != -1:
                chunk[1] = idx
            else:
                chunk = [idx, idx]
            if idx == len(pred_labels) - 1:
                chunks.append(chunk)
        elif label == 'O':
            if chunk[1] != -1:
                chunks.append(chunk)
            chunk = [-1, -1]
    return chunks

def set_seed(seed=719):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True