"""
Written by Romain Froger
Last update: 2024-04-23

This script is used to clean the dataset of the propaganda dataset
by creating an adapted json file for the hateful memes challenge.
Written in the context of CS7643 Deep Learning project at Georgia Tech.
"""

import json
import os

# Load the data from train_v1.jsonl, val_v1.jsonl, test_v1.jsonl
og_data_paths = ["train_v1.jsonl", "val_v1.jsonl", "test_v1.jsonl"]
data = []
for og_data_path in og_data_paths:
    data += [json.loads(line) for line in open(og_data_path, 'r')]

# Only keep examples labeled as "very harmful" (labels are lists)
data = [d for d in data if "very harmful" in d["labels"]]
print(f"Number of examples labeled as 'very_harmful': {len(data)}")

# remove all images from the img folder that are not in the dataset
deleted_img_count = 0
img_ids = [d["image"] for d in data]
img_folder = "img"
for img in os.listdir(img_folder):
    if img not in img_ids:
        try:
            os.remove(os.path.join(img_folder, img))
        except Exception as e:
            print(f"Error deleting {img}: {e}")
        deleted_img_count += 1

print(f"Deleted {deleted_img_count} images from the img folder")

# Create a new json file with the cleaned data called propaganda.jsonl
new_data_path = "propaganda.jsonl"
with open(new_data_path, 'w') as f:
    for d in data:
        img_id = d["image"]
        d["img"] = f"img/{img_id}"
        d["label"] = 1
        del d["labels"]
        del d["image"]
        d = {k: d[k] for k in ["id", "img", "label", "text"]}
        f.write(json.dumps(d) + "\n")

