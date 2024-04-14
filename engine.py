import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch.nn as nn

def train_one_epoch(model, optimizer, train_loader, device, criterion):
    model.train()
    model.to(device)

    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (imgs, texts, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(imgs, texts)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item()}")

    avg_loss = total_loss / num_batches
    print(f"Average Loss: {avg_loss}")

    return avg_loss


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
            loss = criterion(output.squeeze(), labels.float())

            total_loss += loss.item()

            # Compute predictions and targets
            predicted_labels = torch.round(torch.sigmoid(output))
            predictions.extend(predicted_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss}")

    # Compute accuracy, F1 score, and AUROC
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    auroc = roc_auc_score(targets, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"AUROC: {auroc}")

    return avg_loss, accuracy, f1, auroc

    return avg_loss