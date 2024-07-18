import numpy as np
import os
import random
import pandas as pd
import tensorflow as tf
import tqdm
import pickle

def get_images_from_df(df, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        if len(df) < nb_samples:
            nb_samples = len(df)
        df = df.sample(n=nb_samples)
    images = list(zip(df['label'].tolist(), df['image_path'].tolist()))
    if shuffle:
        random.shuffle(images)
    return images

class DataGenerator:
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, nway, kshot, kquery, meta_batchsz, total_batch_num=200000):
        """
        :param nway:
        :param kshot:
        :param kquery:
        :param meta_batchsz:
        """
        self.meta_batchsz = meta_batchsz
        # number of images to sample per class
        self.nimg = kshot + kquery
        self.nway = nway
        self.imgsz = (84, 84)
        self.total_batch_num = total_batch_num
        self.dim_input = np.prod(self.imgsz) * 3  # 21168
        self.dim_output = nway

        base_dir = r'C:\Users\jenis\Downloads\ab\miniimagenet'
        metatrain_csv = os.path.join(base_dir, 'train.csv')
        metaval_csv = os.path.join(base_dir, 'val.csv')
        metatest_csv = os.path.join(base_dir, 'test.csv')

        self.metatrain_data = pd.read_csv(metatrain_csv)
        self.metaval_data = pd.read_csv(metaval_csv)
        self.metatest_data = pd.read_csv(metatest_csv)
        
        self.rotations = [0]

        print('metatrain data sample:', self.metatrain_data.head())
        print('metaval data sample:', self.metaval_data.head())

    def make_data_tensor(self, training=True):
        if training:
            data = self.metatrain_data
            num_total_batches = self.total_batch_num
        else:
            data = self.metaval_data
            num_total_batches = 600

        regenerate = False

        if training and os.path.exists('filelist.pkl'):
            try:
                with open('filelist.pkl', 'rb') as f:
                    filelist = pickle.load(f)
                    all_filenames = filelist['filenames']
                    all_labels = filelist['labels']
                    print('load episodes from file, len:', len(all_filenames))
            except (EOFError, pickle.UnpicklingError, KeyError):
                print("Filelist is corrupted or missing required keys, regenerating...")
                regenerate = True
        else:
            regenerate = True

        if regenerate:
            all_filenames = []
            all_labels = []
            for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                available_classes = len(data['label'].unique())
                if available_classes < self.nway:
                    print(f"Warning: Only {available_classes} unique classes available, adjusting nway to {available_classes}.")
                    sampled_data = data.sample(n=available_classes, replace=True)
                else:
                    sampled_data = data.sample(n=self.nway, replace=True)
                
                labels_and_images = get_images_from_df(sampled_data, nb_samples=self.nimg, shuffle=False)

                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
                all_labels.extend(labels)

            if training:
                with open('filelist.pkl', 'wb') as f:
                    pickle.dump({'filenames': all_filenames, 'labels': all_labels}, f)
                    print('save all file list to filelist.pkl')

        print('creating pipeline ops')

        def _parse_function(filename):
            filename = tf.strings.regex_replace(filename, '\\\\', '/')
            image_string = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.resize(image, [self.imgsz[0], self.imgsz[1]])
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            return image

        # Normalize paths and print them
        valid_filenames = []
        valid_labels = []
        for fname, label in zip(all_filenames, all_labels):
            norm_path = os.path.normpath(os.path.join('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet', fname))
            if os.path.exists(norm_path):
                valid_filenames.append(norm_path)
                valid_labels.append(label)
            else:
                print(f"Missing file: {norm_path}")

        print("Valid filenames:")
        for vf in valid_filenames[:10]:  # Print first 10 valid filenames for review
            print(vf)

        if len(valid_filenames) < len(all_filenames):
            print(f"Found {len(all_filenames) - len(valid_filenames)} missing files. Only using valid files.")

        if len(valid_filenames) == 0:
            raise ValueError("No valid image files found. Please check the dataset paths.")

        dataset = tf.data.Dataset.from_tensor_slices((valid_filenames, valid_labels))
        dataset = dataset.map(lambda filename, label: (_parse_function(filename), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.meta_batchsz * self.nway * self.nimg)

        iterator = iter(dataset)
        images, labels = next(iterator)

        all_image_batches, all_label_batches = [], []
        print('manipulating images to be right order')
        for i in range(self.meta_batchsz):
            image_batch = images[i * (self.nway * self.nimg):(i + 1) * (self.nway * self.nimg)]
            label_batch = labels[i * (self.nway * self.nimg):(i + 1) * (self.nway * self.nimg)]
            new_list, new_label_list = [], []
            for k in range(self.nimg):
                class_idxs = tf.range(0, self.nway)
                class_idxs = tf.random.shuffle(class_idxs)
                true_idxs = class_idxs % self.nway + k * self.nway  # Ensure indices are within range

                # Debug print for indices and labels
                print(f"class_idxs: {class_idxs}")
                print(f"true_idxs: {true_idxs}")

                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))

            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)

        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.nway)

        print('image_b:', all_image_batches)
        print('label_onehot_b:', all_label_batches)

        return all_image_batches, all_label_batches

    def make_test_data_tensor(self):
        data = self.metaval_data
        num_total_batches = 600
        all_filenames = []
        all_labels = []

        for _ in tqdm.tqdm(range(num_total_batches), 'generating test episodes'):
            available_classes = len(data['label'].unique())
            if available_classes < self.nway:
                print(f"Warning: Only {available_classes} unique classes available, adjusting nway to {available_classes}.")
                sampled_data = data.sample(n=available_classes, replace=True)
            else:
                sampled_data = data.sample(n=self.nway, replace=True)
                
            labels_and_images = get_images_from_df(sampled_data, nb_samples=self.nimg, shuffle=False)

            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)
            all_labels.extend(labels)

        print('creating test pipeline ops')

        def _parse_function(filename):
            filename = tf.strings.regex_replace(filename, '\\\\', '/')
            image_string = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.resize(image, [self.imgsz[0], self.imgsz[1]])
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            return image

        # Normalize paths and print them
        valid_filenames = []
        valid_labels = []
        for fname, label in zip(all_filenames, all_labels):
            norm_path = os.path.normpath(os.path.join('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet', fname))
            if os.path.exists(norm_path):
                valid_filenames.append(norm_path)
                valid_labels.append(label)
            else:
                print(f"Missing file: {norm_path}")

        print("Valid filenames:")
        for vf in valid_filenames[:10]:  # Print first 10 valid filenames for review
            print(vf)

        if len(valid_filenames) < len(all_filenames):
            print(f"Found {len(all_filenames) - len(valid_filenames)} missing files. Only using valid files.")

        if len(valid_filenames) == 0:
            raise ValueError("No valid image files found. Please check the dataset paths.")

        dataset = tf.data.Dataset.from_tensor_slices((valid_filenames, valid_labels))
        dataset = dataset.map(lambda filename, label: (_parse_function(filename), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.meta_batchsz * self.nway * self.nimg)

        iterator = iter(dataset)
        images, labels = next(iterator)

        all_image_batches, all_label_batches = [], []
        print('manipulating images to be right order')
        for i in range(self.meta_batchsz):
            image_batch = images[i * (self.nway * self.nimg):(i + 1) * (self.nway * self.nimg)]
            label_batch = labels[i * (self.nway * self.nimg):(i + 1) * (self.nway * self.nimg)]
            new_list, new_label_list = [], []
            for k in range(self.nimg):
                class_idxs = tf.range(0, self.nway)
                class_idxs = tf.random.shuffle(class_idxs)
                true_idxs = class_idxs % self.nway + k * self.nway  # Ensure indices are within range

                # Debug print for indices and labels
                print(f"class_idxs: {class_idxs}")
                print(f"true_idxs: {true_idxs}")

                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))

            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)

        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.nway)

        print('image_b:', all_image_batches)
        print('label_onehot_b:', all_label_batches)

        return all_image_batches, all_label_batches
