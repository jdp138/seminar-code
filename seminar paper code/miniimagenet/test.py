import os
import pandas as pd
import random

# Define paths (Assuming 'miniimagenet' directory structure)
image_dir = 'C:\\Users\\jenis\\Downloads\\ab\\miniimagenet'
healthy_dir = os.path.join(image_dir, 'healthy')
diseased_dir = os.path.join(image_dir, 'diseased')

# Get list of all images in both directories
healthy_images = [os.path.join('healthy', img) for img in os.listdir(healthy_dir) if img.endswith('.JPG')]
diseased_images = [os.path.join('diseased', img) for img in os.listdir(diseased_dir) if img.endswith('.JPG')]

# Combine all images and create labels
all_images = healthy_images + diseased_images
labels = [0] * len(healthy_images) + [1] * len(diseased_images)

# Shuffle the dataset
combined = list(zip(all_images, labels))
random.shuffle(combined)
all_images[:], labels[:] = zip(*combined)

# Split the data
train_split = int(0.7 * len(all_images))
val_split = int(0.85 * len(all_images))

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

train_labels = labels[:train_split]
val_labels = labels[train_split:val_split]
test_labels = labels[val_split:]

# Create DataFrames
train_df = pd.DataFrame({'image_path': train_images, 'label': train_labels})
val_df = pd.DataFrame({'image_path': val_images, 'label': val_labels})
test_df = pd.DataFrame({'image_path': test_images, 'label': test_labels})

# Save to CSV
train_df.to_csv('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet\\train.csv', index=False)
val_df.to_csv('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet\\val.csv', index=False)
test_df.to_csv('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet\\test.csv', index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Training Set", dataframe=train_df)
tools.display_dataframe_to_user(name="Validation Set", dataframe=val_df)
tools.display_dataframe_to_user(name="Test Set", dataframe=test_df)

train_df.head(), val_df.head(), test_df.head()
