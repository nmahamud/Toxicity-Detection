# TEAM NAME: Monkeys
# TEAM MEMBERS: Aawab Mahmood, Nazif Mahamud, Kevin Wei

import torch
from torch.utils.data import Dataset, dataloader
from transformers import RobertaTokenizerFast
from collections import Counter
import re
import csv

NUM_CLASSES = 6
NUM_SENTIMENT = 3
NUM_HATE = 2

BATCH_SIZE=32
EPOCHS=1

trainDatasetPath = './train.csv'

class ToxicCommentsRobertaDataset(Dataset):
    def __init__(self, path, columnName, tokenizer, maxLength=512):
        self.texts = []
        self.labels_list = []
        self.tokenizer = tokenizer
        self.maxLength = maxLength

        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            textColIndex = header.index(columnName)
            labelColIndices = [
                header.index(label)
                for label in [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                ]
            ]
            printed = False
            for rowIndex, row in enumerate(reader):
                text = row[textColIndex]
                # Store label columns independently
                labels = [int(row[labelColIndex]) for labelColIndex in labelColIndices] 

                self.texts.append(text)
                self.labels_list.append(labels)

        self.labels = torch.tensor(self.labels_list, dtype=torch.float32) 

        # Calculate weights independently per label
        pos_labels = self.labels.sum(dim=0)
        neg_labels = self.labels.shape[0] - pos_labels 
        self.weights = neg_labels/pos_labels 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int):
        text = self.texts[index]
        labels = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.maxLength,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels,
        }

class ToxicCommentsRobertaEvalDataset(Dataset):
    def __init__(self, df, tokenizer, maxLength=512):
        self.df = df
        self.tokenizer = tokenizer
        self.maxLength = maxLength

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['comment_text']
        labels = self.df.iloc[index][['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(int)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.maxLength,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }