import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertClsModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClsModel, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, 
                input_ids, 
                token_type_ids=None, 
                attention_mask=None):
        
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            return_dict=False)
        
        pooled_output = nn.functional.relu(self.hidden(self.dropout(pooled_output)))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits