import torch
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


def evaluate():
    pass