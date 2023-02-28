import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import Dataset
    
class ClsDataset(Dataset):
    def __init__(self, data_path, label2id, max_seq_len, tokenizer, input_text1=None, input_text2=None, read_from_file=True):
        if read_from_file: self.examples = self._read_tsv(data_path)
        else: self.examples = self._read_test_data(input_text1, input_text2)
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
    def _read_test_data(input_text1, input_text2):
        all_samples = []
        for text1, text2 in zip(input_text1, input_text2):
            all_samples.append({'text1': text1, 'text2': text2, 'label': '0'})
        return all_samples
    
    @staticmethod
    def _read_tsv(data_path):
        all_samples = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                text1, text2, label = line.strip().split('\t')
                all_samples.append({'text1': text1, 'text2': text2, 'label': label})
        return all_samples
    
    @staticmethod
    def convert_example_to_feature(example, label2id, max_seq_len, tokenizer):
        features = tokenizer.encode_plus(
            example["text1"],
            text_pair=example["text2"],
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
        )
        input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(features["token_type_ids"], dtype=torch.long)
        label_ids = torch.tensor(label2id[example['label']], dtype=torch.long)
        return input_ids, token_type_ids, label_ids
    
class ClsMetrics(object):
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
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1
    
    def result(self):
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'Precision': precision, 'Recall': recall, 'F1': f1}
    
    def update(self, true_labels, pred_labels):
        self.origins.extend(true_labels)
        self.founds.extend(pred_labels)
        self.rights.extend([pred_label for pred_label, true_label in zip(pred_labels, true_labels) if pred_label == true_label])

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