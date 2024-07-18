# MAML-TensorFlow

# Highlights
- adopted from cbfin's official implementation with equivalent performance on mini-imagenet
- clean, tiny code style and very easy-to-follow from comments almost every lines
- faster and trivial improvements, eg. 0.335s per epoch comparing with 0.563s per epoch, saving up to **3.8 hours** for total 60,000 training process

# dependencies 
- numpy
- tenserflow
- pandas
- pickle
- tqdm
- pIL


# How-TO
1. replace the `path` by your actual path in `data_generator.py`:
```python
		base_dir = r'C:\Users\jenis\Downloads\ab\miniimagenet'
		norm_path = os.path.normpath(os.path.join('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet', fname))
		norm_path = os.path.normpath(os.path.join('C:\\Users\\jenis\\Downloads\\ab\\miniimagenet', fname))	
```
Also replace the `path` by your actual path in `proc_images.py`:
```python
		path_to_images = r'C:\Users\jenis\Downloads\ab\miniimagenet\images'
		base_dir = 'C:\\Users\\jenis\\Downloads\\ab\\miniimagenet'
```

2. resize all raw images to 84x84 size by
```shell
python proc_images.py
```

3. train
```shell
python main.py
```

and then minitor training process by:
```shell
tensorboard --logdir logs
```

4. test
```shell
python main.py --test
```

As MAML need generate 200,000 train/eval episodes before training, which usually takes up to 6~8 minutes, I use an cache file `filelist.pkl` to dump all these episodes for the first time and then next time the program will load from the cached file. It only takes several seconds to load from cached files.

generating episodes: 100%|█████████████████████████████████████████| 200000/200000 [04:38<00:00, 717.27it/s]

