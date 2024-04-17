import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from typing import Iterable

def train_one_epoch(model: torch.nn.Module, criterion:torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch=int, max_grad_norm: float = 0.1):
    model.train()
    model.to(device)

    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, (imgs, texts, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(imgs, texts)
        labels = torch.stack([1-labels, labels], dim=1)
        loss = criterion(output.squeeze().float() , labels.float())

        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch[{epoch}] [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f} ({total_loss / (batch_idx + 1):.4f})")

    avg_loss = total_loss / num_batches

    return {"loss": avg_loss}


def evaluate(model, val_loader, device, criterion):
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = len(val_loader)

    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, (imgs, texts, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            output = model(imgs, texts)
            labels = torch.stack([1-labels, labels], dim=1)
            loss = criterion(output.squeeze().float() , labels.float())

            total_loss += loss.item()

            # Compute predictions and targets
            predicted_labels = torch.argmax(output, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Compute accuracy, F1 score, and AUROC
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    auroc = roc_auc_score(targets, predictions)

    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "auroc": auroc}
