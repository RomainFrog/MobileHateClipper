import torch
import torch.nn as nn
import pickle

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
        loss = criterion(output.squeeze(), labels.float())

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
    
    fim_TP, fim_FP, fim_FN, fim_TN = None, None, None, None

    TP, FP, FN, TN = 0, 0, 0, 0
    fim_dict = {'TP': None, 'FP': None, 'FN': None, 'TN': None}
    l_fim_dict = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    count_dict = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    print(model.fusion)

    with torch.no_grad():
        for batch_idx, (imgs, texts, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            output, fim = model(imgs, texts)
            loss = criterion(output.squeeze(), labels.float())

            total_loss += loss.item()

            # Compute predictions and targets
            predicted_labels = torch.round(torch.sigmoid(output))
            predicted_labels = predicted_labels.cpu().numpy()            
            labels = labels.cpu().numpy()
            predictions.extend(predicted_labels)
            targets.extend(labels)
            
            # update the mean FIM by iterating on the batch  
            fim = fim.cpu().numpy()
            

            for i in range(labels.size):
                predicted_label = predicted_labels[i]
                label = labels[i]
                key = 'TP' if predicted_label == 1 and label == 1 else \
                    'FP' if predicted_label == 1 and label == 0 else \
                    'FN' if predicted_label == 0 and label == 1 else \
                    'TN'
                count_dict[key] += 1
                if model.fusion == 'align':
                    if fim_dict[key] is None:
                        fim_dict[key] = fim[i]
                    else:
                        fim_dict[key] += fim[i]
                
                if model.fusion == 'align':
                    l_fim_dict[key].append(fim[i])
            
    avg_loss = total_loss / num_batches

    # Compute accuracy, F1 score, and AUROC
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    auroc = roc_auc_score(targets, predictions)

   

    # save as a pickle file
    if model.fusion == "cross":
        # Compute the mean FIM
        fim_TP = fim_TP / TP
        fim_FP = fim_FP / FP
        fim_FN = fim_FN / FN
        fim_TN = fim_TN / TN
        with open(f"fim.pkl", "wb") as f:
            pickle.dump(fim_dict, f)

    if model.fusion == "align":
        with open(f"l_fim.pkl", "wb") as f:
            pickle.dump(l_fim_dict, f)


    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "auroc": auroc}
