import torch
from torch.utils.data import Dataset, dataloader
from transformers import RobertaTokenizerFast
from collections import Counter
import re
import csv

trainDatasetPath = './jigsaw-toxic-comment-classification-challenge/train.csv'


class ToxicCommentsRobertaDataset(Dataset):
    def __init__(self, path, columnName, tokenizer, maxLength=512):
        """
        Args:
            path (string): Path to the csv file with annotations.
            columnName (string): Name of the column containing the comments.
            tokenizer: Tokenizer to use for tokenizing the comments.
            maxLength (int): Maximum length of the tokenized comments.
        """
        self.texts = []
        self.labels_list = []
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            textColIndex = header.index(columnName)
            labelColIndices = [header.index(label) for label in ['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
            for rowIndex, row in enumerate(reader):
                text = row[textColIndex]
                labels = [row[labelColIndex] for labelColIndex in labelColIndices]
                self.texts.append(text)
                self.labels_list.append(labels)
        self.labels = torch.tensor(self.labels_list, dtype=torch.float32)


    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index: int):
        text = self.texts[index]
        labels = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.maxLength,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# For testing only!
# if __name__ == '__main__':
#     tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
#     toxicDataset = ToxicCommentsRobertaDataset(trainDatasetPath, 'comment_text', tokenizer=tokenizer)
    
#     if len(toxicDataset) > 0:
#         print(f"\nLength of dataset: {len(toxicDataset)}")