import json
import os

json_path = "memotion.jsonl"

data = [json.loads(line) for line in open(json_path, 'r')]

# remove images from the /img folder that are not in the json file
img_folder = "img"
deleted_img_count = 0

for img in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img)
    if img_path not in [d['img'] for d in data]:
        os.remove(img_path)
        deleted_img_count += 1

print("Deleted", deleted_img_count, "images from /img folder")
print("Remaining images in /img folder:", len(os.listdir(img_folder)) - deleted_img_count)

