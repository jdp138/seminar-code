from __future__ import print_function
import csv
import glob
import os
import shutil
from PIL import Image

path_to_images = r'C:\Users\jenis\Downloads\ab\miniimagenet\images'

all_images = glob.glob(os.path.join(path_to_images, '*'))

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(f"Resized {i} images")

# Put in correct directory
base_dir = 'C:\\Users\\jenis\\Downloads\\ab\\miniimagenet'
for datatype in ['train', 'val', 'test']:
    datatype_dir = os.path.join(base_dir, datatype)
    os.makedirs(datatype_dir, exist_ok=True)

    with open(os.path.join(base_dir, f'{datatype}.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            image_path = row[0]
            label = row[1]
            image_name = os.path.basename(image_path)
            if label != last_label:
                cur_dir = os.path.join(datatype_dir, label)
                os.makedirs(cur_dir, exist_ok=True)
                last_label = label
            src_path = os.path.join(path_to_images, image_name)
            dst_path = os.path.join(cur_dir, image_name)
            print(f"Moving {src_path} to {dst_path}")
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"File not found: {src_path}")
