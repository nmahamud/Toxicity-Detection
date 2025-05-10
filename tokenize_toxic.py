import torch
from torch.utils.data import Dataset, dataloader
from transformers import RobertaTokenizerFast
from collections import Counter
import re
import csv

trainDatasetPath = './jigsaw-toxic-comment-classification-challenge/train.csv'

class LazyCSVDataset(Dataset):
    def __init__(self, file_path, tokenizer, text_column_index, label_column_indices, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.text_column_index = text_column_index
        self.label_column_indices = label_column_indices
        self.max_length = max_length
        self.file_path = file_path
        self.offsets = []
        self.num_rows = 0

        # Build index of line offsets
        with open(file_path, 'r') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                self.num_rows += 1
                offset += len(line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            f.seek(self.offsets[idx])
            reader = csv.reader(f)
            next(reader)  # Skip header
            line = next(reader)
            text = line[self.text_column_index]
            labels = [float(line[i]) for i in self.label_column_indices]
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(labels, dtype=torch.float32)
            }

# if __name__ == '__main__':
#     tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
#     text_column_index = 1
#     label_column_indices = [2, 3, 4, 5, 6, 7]
#     max_length = 512
#     dataset = LazyCSVDataset(trainDatasetPath, tokenizer, text_column_index, label_column_indices, max_length)
#     print(dataset[1]['input_ids'])
    