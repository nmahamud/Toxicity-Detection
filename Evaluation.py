import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import Constants
import trim from Training.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model(model, df, tokenizer, device='cuda'):
    model.to(device)
    model.eval()

    preds = []
    true_labels = []

    with torch.no_grad():
        for index in range(len(df)):
            text = df.iloc[index]['comment_text']
            label = df.iloc[index]['labels']

            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in encoding.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs).squeeze()

            # Sigmoid act so just check if >0.5
            pred = (outputs > 0.5).long()

            preds.append(pred.cpu().item())
            true_labels.append(label)

    f1 = sklearn.metrics.f1_score(true_labels, preds, average='binary')
    accuracy = sklearn.metrics.accuracy_score(true_labels, preds)
    precision = sklearn.metrics.precision_score(true_labels, preds)
    recall = sklearn.metrics.recall_score(true_labels, preds)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    return metrics


def visualize_metrics(df_filtered, metrics, loss_per_epoch, true_labels, preds):
    '''
        Visualizes all metrics after training
    '''
    plt.figure(figsize=(8, 6))
    sns.countplot(x='labels', data=df_filtered)
    plt.title('Distribution of Labels in Test Data')
    plt.xlabel('Label (0: Non-Toxic, 1: Toxic)')
    plt.ylabel('Count')
    plt.show()

    # 2. Metrics Visualization
    metrics_df = pd.DataFrame([metrics])
    metrics_df = metrics_df.T.reset_index()
    metrics_df.columns = ['Metric', 'Value']


    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', data=metrics_df)
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1) # Assuming metrics are between 0 and 1
    plt.ylabel('Score')
    plt.show()


    # 3. Confusion Matrix (requires predictions and true labels)

    # Assuming 'preds' and 'true_labels' were saved during evaluation.
    # Replace these with the actual predictions and true labels from evaluate_model
    # preds = ...
    # true_labels = ...
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Toxic', 'Predicted Toxic'],
                yticklabels=['Actual Non-Toxic', 'Actual Toxic'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 4.  Loss Curve (if you saved the loss_per_epoch during training)

    plt.figure(figsize=(10, 6))
    for epoch_losses in loss_per_epoch:
        plt.plot(range(len(epoch_losses)), epoch_losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()