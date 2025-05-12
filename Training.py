# TEAM NAME: Monkeys
# TEAM MEMBERS: Aawab Mahmood, Nazif Mahamud, Kevin Wei

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 6
NUM_SENTIMENT = 3
NUM_HATE = 2

BATCH_SIZE=32
EPOCHS=1

def trim(batch: dict[str, torch.tensor]):
    masks = batch['attention_mask']
    maxLen = torch.max(torch.sum(masks, dim=1))

    if batch['labels'].shape == batch['input_ids'].shape:
        batch['labels'] = batch['labels'][:, :maxLen]

    batch['input_ids'] = batch['input_ids'][:, :maxLen]
    batch['attention_mask'] = batch['attention_mask'][:, :maxLen]

    return batch

def train_model(model, dataset: Dataset, lr=1e-5, weight_decay=1e-3):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(lr=lr, weight_decay=weight_decay, params=model.parameters())


    trainDataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    lossPerEpoch = []

    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for batch in trainDataLoader:
            batch = trim(batch)

            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            labels = batch["labels"].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs).squeeze()

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        lossPerEpoch.append(losses)

    return lossPerEpoch