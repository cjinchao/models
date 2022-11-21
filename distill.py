import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchtext.datasets import DBpedia
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer

parser = argparse.ArgumentParser(description='Distill')
parser.add_argument('--teacher', type=str, default='xlm-roberta-base', help='teacher model name or model config')
parser.add_argument('--student', type=str, default='/Users/caijinchao/Desktop/xlm-roberta-tiny', help='student model name or model config')
parser.add_argument('--data', type=str, default='data', help='data path')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--temperature', type=float, default=2, help='temperature')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--seed', type=int, default=42, help='random seed')

def encode_and_batchify(batch, tokenizer, max_seq_len=128, label_map=None):
    batch_size = len(batch)
    labels = []
    input_ids = []
    attention_mask = []
    for i in range(batch_size):
        inputs = tokenizer.encode_plus(batch[i][1], add_special_tokens=True)
        example_len = len(inputs['input_ids'])
        if example_len > max_seq_len:
            inputs['input_ids'] = inputs['input_ids'][:max_seq_len]
            inputs['attention_mask'] = inputs['attention_mask'][:max_seq_len]
        else:
            inputs['input_ids'].extend([tokenizer.pad_token_id] * (max_seq_len - example_len))
            inputs['attention_mask'].extend([0] * (max_seq_len - example_len))
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
        if label_map is None:
            labels.append(batch[i][0])
        else:
            labels.append(label_map[batch[i][0]])
    return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask), 'labels': torch.tensor(labels)}


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    teacher = XLMRobertaForSequenceClassification.from_pretrained(args.teacher, num_labels=14)
    student_config = XLMRobertaConfig.from_pretrained(args.student, num_labels=14)
    student = XLMRobertaForSequenceClassification(student_config)

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.teacher)
    
    train_iter = DBpedia(split='train')
    # test_iter = DBpedia(split='test')

    dataloader = DataLoader(train_iter, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: encode_and_batchify(x, tokenizer, label_map={i+1:i for i in range(14)}))
    # define loss
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    ce_critetion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        # train
        student.train()
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # teacher outputs
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
                teacher_probs = F.softmax(teacher_logits / args.temperature, dim=-1)

            # student outputs
            student_outputs = student(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits
            student_probs = F.softmax(student_logits / args.temperature, dim=-1)

            # distillation loss
            print(student_probs.size(), teacher_probs.size(), labels.size())
            loss = args.alpha * ce_critetion(student_probs, labels) + (1 - args.alpha) * kl_criterion(student_probs, teacher_probs)
            loss.backward()
            optimizer.step()
            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.cpu().item()))

if __name__ == '__main__':
    main()