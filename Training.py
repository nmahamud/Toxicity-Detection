import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import Constants


def trim(batch: dict[str, torch.tensor]):
    masks = batch['attention_mask']
    max_len = torch.max(torch.sum(masks, dim=1))
    
    if batch['labels'].shape == batch['input_ids'].shape:
        batch['labels'] = batch['labels'][:, :max_len]

    batch['input_ids'] = batch['input_ids'][:, :max_len]
    batch['attention_mask'] = batch['attention_mask'][:, :max_len]
    
    return batch

def train_model(model, dataset: Dataset, lr=1e-5, weight_decay=1e-3):
    optimizer = torch.optim.AdamW(lr=lr, weight_decay=weight_decay, params=model.parameters())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_dataloader = DataLoader(dataset, batch_size=Constants.batch_size, drop_last=True, shuffle=True)
    loss_per_epoch = []

    criterion = torch.nn.BCELoss()
    
    model.train()
    for epoch in Constants.epochs:
        losses = []
        for batch in train_dataloader:
            batch = trim(batch)

            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs).squeeze()
            
            labels = batch["labels"]
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_per_epoch.append(losses)

    return loss_per_epoch

    

