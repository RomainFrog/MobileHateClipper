"""
This file contains the code to evaluate the ensemble model
which is a combination of all best models from different
configurations.

Code written by: Romain Froger
Last modified: 2024-28-04

Written for the project MobileHateClipper for CS7641 at Georgia Tech.
"""

import os
import torch
import mobileclip
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from models_mobile_hate_clipper import MobileHateClipper
from datasets import HatefulMemesDataset


# There is a folder "outputs" containing subfolders for each configuration
# Each subfolder contains the best model for this configuration with the following name:
# "checkpoint-best-<epoch>.pth"
# We will load all these models, predict the test set and compute the ensemble prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HatefulMemesDataset('data/hateful_memes/', 'data/hateful_memes/', split='test_seen', clip_model='mobileclip_b', use_propaganda=True, use_memotion=True)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

checkpoint_path = os.path.join('checkpoints', 'mobileclip_b.pt')
clip_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_b', pretrained=checkpoint_path)


# List all subfolders in the "outputs" folder
output_dir = 'outputs'
subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir()]

preds = []

for subfolder in subfolders:
    # the name of the subfolder is <FUSION>_ed<EMBED_SIZE>_nml<NUM_PROJ_LAYERS>_npol<NUM_PRE_OUT_LAYERS>

    fusion = subfolder.split('_')[0][8:]
    embed_size = int(subfolder.split('ed')[1].split('_')[0])
    num_proj_layers = int(subfolder.split('nml')[1].split('_')[0])
    num_pre_out_layers = int(subfolder.split('npol')[1])

    # checkpoint is named "checkpoint-best-<epoch>.pth"
    checkpoint_file = [f for f in os.listdir(subfolder) if f.startswith('checkpoint-best')][0]

    print(f'Fusion: {fusion}, Embedding size: {embed_size}, Number of projection layers: {num_proj_layers}, Number of pre-output layers: {num_pre_out_layers}, Checkpoint file: {checkpoint_file}')
    # Load the model
    model = MobileHateClipper(fusion=fusion, embed_dim = embed_size, pre_output_dim=512,
                num_mapping_layers = num_proj_layers, num_pre_output_layers=num_pre_out_layers, clip_model=clip_model,
                dropout_rates=[0, 0, 0], freeze_clip=True)
    
    model.load_state_dict(torch.load(os.path.join(subfolder, checkpoint_file)))
    model.eval()
    model.to(device)

    predictions = []
    targets = []


    with torch.no_grad():
        for batch_idx, (imgs, texts, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            output = model(imgs, texts)

            # Compute predictions and targets
            predicted_labels = torch.round(torch.sigmoid(output))
            predictions.extend(predicted_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

        accuracy = accuracy_score(targets, predictions)
        auroc = roc_auc_score(targets, predictions)
        print(f'Accuracy: {accuracy}, AUROC: {auroc}')

    preds.append(predictions)

# save the predictions
np.save('ensemble_preds.npy', preds)
# save the targets
np.save('ensemble_targets.npy', targets)

# Compute the ensemble prediction
preds = np.array(preds)
preds = preds.squeeze() 
# majority vote
ensemble_preds = np.round(np.mean(preds, axis=0))


accuracy = accuracy_score(targets, ensemble_preds)
auroc = roc_auc_score(targets, ensemble_preds)

print(f'Accuracy: {accuracy}, AUROC: {auroc}')


