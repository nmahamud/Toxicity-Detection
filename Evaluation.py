# TEAM NAME: Monkeys
# TEAM MEMBERS: Aawab Mahmood, Nazif Mahamud, Kevin Wei

# Description: This code file houses our main evaluation and metric/graph/figure printing functions
# System: Compiled into the .ipynb and ran on our Google Colab instances and local PCs

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import trim from Training.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

NUM_CLASSES = 6
NUM_SENTIMENT = 3
NUM_HATE = 2

BATCH_SIZE=32
EPOCHS=1

def evalModel(model, dataset: ToxicCommentsRobertaEvalDataset, device='cuda'):
    model.to(device)
    model.eval()

    evalDataLoader = DataLoader(dataset, batch_size=BATCH_SIZE)

    preds = []
    trueLabels = []

    with torch.no_grad():
        for batch in evalDataLoader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs).squeeze()

            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).int().cpu().numpy()

            preds.extend(pred)  
            trueLabels.extend(labels.cpu().numpy())  

    preds = np.array(preds)
    trueLabels = np.array(trueLabels) 

    print(preds.shape)
    print(trueLabels.shape)

    cm = multilabel_confusion_matrix(trueLabels, preds)

    f1 = sklearn.metrics.f1_score(trueLabels, preds, average='macro')
    accuracy = sklearn.metrics.accuracy_score(trueLabels, preds)
    precision = sklearn.metrics.precision_score(trueLabels, preds, average='macro')
    recall = sklearn.metrics.recall_score(trueLabels, preds, average='macro')

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    return metrics, preds, trueLabels, cm

def visualize(dfFiltered, metrics, lossPerEpoch, trueLabels, preds, cm):
    dfMelted = dfFiltered.melt(
        id_vars=['id', 'comment_text'],
        value_vars=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        var_name='labels',
        value_name='label_value' 
    )

    dfMelted = dfMelted[dfMelted['label_value'] != -1]

    plt.figure(figsize=(8, 6))
    sns.countplot(x='labels', data=dfMelted[dfMelted['label_value'] == 1])
    plt.title('Distribution of Labels in Test Data')
    plt.xlabel('Label (0: Non-Toxic, 1: Toxic)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  
    plt.show()


    labelNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] 
    f1Scores = sklearn.metrics.f1_score(trueLabels, preds, average=None)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labelNames, y=f1Scores)
    plt.title('F1 Score per Label')
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right') 
    plt.show()


    labelNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    numLabels = len(labelNames)

    cm_all = np.zeros((2, 2), dtype=int) 
    for i in range(numLabels):
        cm_all += cm[i] 

    plt.figure(figsize=(6, 4)) 
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Overall Confusion Matrix (All Labels)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    plt.figure(figsize=(10, 6))
    for losses in lossPerEpoch:
        plt.plot(range(len(losses)), losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()