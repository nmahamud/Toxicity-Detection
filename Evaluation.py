import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import Constants
import trim from Training.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def evaluate_model(model, dataset: Dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=Constants.batch_size, drop_last=True, shuffle=False)
    
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = trim(batch)

            batch_labels = batch.pop('labels')
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs).squeeze()
            
            # Sigmoid act so just check if >0.5
            pred = (outputs > 0.5).long()
            
            # Store pred and true_labels
            preds.extend(pred.cpu().numpy())
            true_labels.extend(batch_labels.numpy())
    
    
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