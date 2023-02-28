import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from .utils import ClsDataset, ClsMetrics
from .utils import set_seed
from .model import BertClsModel

logger = logging.getLogger()

def train():
    config = BertConfig.from_pretrained(model_name, num_labels=len(label2id))
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertClsModel.from_pretrained(model_name, config=config)
    model.to(device)
    train_ds = ClsDataset(data_path=train_path, label2id=label2id, max_seq_len=max_seq_len, tokenizer=tokenizer)
    dev_ds = ClsDataset(data_path=dev_path, label2id=label2id, max_seq_len=max_seq_len, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    num_training_steps = len(train_loader) * num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(num_training_steps * warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    loss_fct = nn.CrossEntropyLoss()
    metric = ClsMetrics(id2label)
    
    global_step, best_f1 = 1, 0.
    eval_step = len(train_loader)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    for epoch in range(1, num_epoch + 1):
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = loss_fct(logits, labels) # [batch_size, max_seq_len, num_labels]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_step > 0 and global_step % log_step == 0:
                print(f'epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss: {loss.item():.6f}')
            if global_step > 0 and global_step % eval_step == 0:
                results = evaluate(model, dev_loader, metric)
                model.train()
                tmp_f1 = results['F1']
                if tmp_f1 > best_f1:
                    print(f'\nBest F1 has been updated: {best_f1:.5f} --> {tmp_f1:.5f}')
                    best_f1 = tmp_f1
                    model_to_save = (model.module if hasattr(model, 'module') else model)
                    model_to_save.save_pretrained(save_path)
                    tokenizer.save_vocabulary(save_path)
                print(f'\nEvaluation results: Precision: {results["Precision"]:.5f}, Recall: {results["Recall"]:.5f}, F1: {results["F1"]:.5f}, Best F1 for now: {best_f1:.5f}')
            global_step += 1
    if 'cuda' in str(device):
        torch.cuda.empty_cache()

def evaluate(model, data_loader, metric):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    metric.reset()
    for batch in tqdm(data_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids=token_type_ids)
        true_labels = labels.cpu().numpy()
        pred_labels = logits.argmax(axis=-1).cpu().numpy() # [batch_size, ]
        metric.update(true_labels=true_labels, pred_labels=pred_labels)
    return metric.result()
    
def predict(model_path, input_text1, input_text2):
    config = BertConfig.from_pretrained(model_path, num_labels=len(label2id))
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertClsModel.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()
    
    if isinstance(input_text1, str) and isinstance(input_text2, str):
        features = tokenizer.encode_plus(
                input_text1,
                text_pair=input_text2,
                max_length=max_seq_len,
                truncation=True,
                return_token_type_ids=True,
            )
        input_ids = torch.tensor(features['input_ids'], dtype=torch.long).unsqueeze(0).to(device) # [batch_size=1, num_labels]
        token_type_ids = torch.tensor(features['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)
        logits = model(input_ids, token_type_ids=token_type_ids)
        pred_label = logits.argmax(axis=-1).cpu().numpy()[0]
        return id2label[pred_label]
    elif isinstance(input_text1, list) and isinstance(input_text2, list):
        test_ds = ClsDataset(data_path=dev_path, label2id=label2id, max_seq_len=max_seq_len, tokenizer=tokenizer, 
                            input_text1=input_text1, input_text2=input_text2, read_from_file=False)
        test_loader = DataLoader(test_ds, batch_size=per_gpu_batch_size, shuffle=False)
        pred_results = []
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, _ = batch
            logits = model(input_ids, token_type_ids=token_type_ids)
            pred_labels = logits.argmax(axis=-1).cpu().numpy().tolist()
            pred_results.extend([id2label[label] for label in pred_labels])
        return pred_results
    else:
        print('input_text1 and input_text2 must be str or list')

    
model_name = '/data/xiancai/cxc/myRepo/pretrains/bert-base-chinese'
train_path = './data/train.tsv'
dev_path = './data/dev.tsv'
save_path = './checkpoint/'

label2id = {'0': 0, '1': 1}
id2label = {0: '0', 1: '1'}
max_seq_len = 128
per_gpu_batch_size = 16
num_epoch = 5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
learning_rate = 5e-5
adam_epsilon = 1e-8
seed = 42
log_step = 100
eval_step = 30

set_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
batch_size = per_gpu_batch_size * n_gpu
    
if __name__ == '__main__':
    # train()
    
    pred_model_path = save_path
    # input_text1 = '若泽·萨尔内的总统是谁？'
    # input_text2 = '若泽·萨尔内总统'
    # input_text1 = '闻一多全集是哪个出版社出版的？'
    # input_text2 = '闻一多全集出版社'
    # input_text1 = '闻一多全集是哪个出版社出版的？'
    # input_text2 = '闻一多全集出版时间'
    input_text1 = ['若泽·萨尔内的总统是谁？', '闻一多全集是哪个出版社出版的？', '闻一多全集是哪个出版社出版的？', ]
    input_text2 = ['若泽·萨尔内总统', '闻一多全集出版社', '闻一多全集出版时间', ]
    pred_results = predict(pred_model_path, input_text1, input_text2)
    print(pred_results)