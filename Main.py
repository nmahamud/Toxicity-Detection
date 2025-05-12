# TEAM NAME: Monkeys
# TEAM MEMBERS: Aawab Mahmood, Nazif Mahamud, Kevin Wei

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast
from collections import Counter
import re
import csv

import sklearn
from torch import nn
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def memStat():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)


if __name__ == '__main__' :
    memStat()
    torch.cuda.empty_cache()
    memStat()
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    toxicDataset = ToxicCommentsRobertaDataset(trainDatasetPath, 'comment_text', tokenizer=tokenizer)

    model = distilRB_sem(multilabel=True)

    lossPerEpoch = trainModel(model, toxicDataset)

    torch.cuda.empty_cache()

    dfTest = pd.read_csv('test.csv', quotechar='"')
    dfLabels = pd.read_csv('test_labels.csv')

    dfMerged = pd.merge(dfTest, dfLabels, on='id', how='left')
    dfFiltered = dfMerged[dfMerged['toxic'] != -1] 

    evalDataset = ToxicCommentsRobertaEvalDataset(dfFiltered, tokenizer)

    # Eval
    metrics, preds, trueLabels, cm  = evalModel(model, evalDataset)
    print(metrics)

    visualize(dfFiltered, metrics, lossPerEpoch, trueLabels, preds, cm) 