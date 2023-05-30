import os
import os.path as p
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
import subprocess

from typing import List, Tuple, Dict, Callable, Optional, Literal

import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import cv2 as cv
from PIL import Image

def train_valid_test_split(array, splits):
    array = np.array(array)
    array = np.sort(array)
    np.random.seed(2022)
    np.random.shuffle(array)
    split_indices = np.cumsum([0] + splits)
    split_indices = (split_indices * len(array)).astype(int)
    split_files = np.split(array, split_indices[1:-1])
    return split_files

def copy_train_valid_test_images(image_files, splits, save_dir, size=256):
    folders = ['train', 'valid', 'test']
    split_files = train_valid_test_split(image_files, splits)
    print(' Split sizes:', [len(split) for split in split_files], ' Total:', len(image_files))
    print(' Copying images...')
    for folder, split in zip(folders, split_files):
        os.makedirs(p.join(save_dir, folder), exist_ok=True)
        for file in tqdm(split):
            try:
                img = cv.imread(file)
                img = cv.resize(img, (size, size))
                # add extension to make sure to convert any format to jpg
                if '.png' in file:
                    file = file.replace('.png', '.jpg')
                else:
                    file = file + '.jpg'
                cv.imwrite(p.join(save_dir, folder, p.basename(file)), img)
            except Exception as e:
                print('Error with file', file, e)
                continue

def make_pad_ufes():
    print('--- PAD UFES 20 ---')
    current_dir = p.dirname(p.realpath(__file__))
    images_folder = current_dir + '/downloaded_data/pad_ufes_20/**/*.png'
    images = glob(images_folder, recursive=True)
    images.sort()
    dataset_metadata = pd.read_csv(current_dir + '/downloaded_data/pad_ufes_20/zr7vgbcyr2-1/metadata.csv')

    print(' Missing', dataset_metadata['fitspatrick'].isna().sum(), 'out of', len(dataset_metadata))

    dataset_metadata = dataset_metadata[dataset_metadata['fitspatrick'].notna()]

    save_dir = current_dir + '/pad_ufes'
    os.makedirs(save_dir, exist_ok=True)

    labels_df = pd.DataFrame({'file_name': dataset_metadata['img_id'], 'label': dataset_metadata['fitspatrick'].astype(int)})
    labels_df.to_csv(p.join(save_dir, 'labels.csv'), index=False)

    # Remove images with missing labels
    images = [i for i in images if p.basename(i) in labels_df['file_name'].values]

    splits = [0.8, 0.1, 0.1]
    copy_train_valid_test_images(images, splits, save_dir)

def download_image(url, file_path, dev_null):
    def run_command():
        cmd = ['wget', '-O', file_path, url]
        subprocess.run(cmd, stdout=dev_null, stderr=dev_null)
    if not p.exists(file_path):
        run_command()
    else:
        # check if corrupted
        try:
            img = Image.open(file_path)
            if img is None or img.size == 0:
                raise Exception('Corrupted image')
            img.verify()
            Image.open(file_path).load()
        except:
            os.remove(file_path)
            run_command()

def download_fp17k_images():
    current_dir = p.dirname(p.realpath(__file__))
    download_dir = current_dir + '/downloaded_data/fp17k/images'
    os.makedirs(download_dir, exist_ok=True)
    
    fp17k_df = pd.read_csv(current_dir + '/downloaded_data/fp17k/fitzpatrick17k.csv')
    fp17k_df = fp17k_df[fp17k_df['url'].notna()]
    urls = fp17k_df['url'].values
    file_names = fp17k_df['md5hash'].values

    with ThreadPoolExecutor(max_workers=8) as executor:
        dev_null = open(os.devnull, 'w')
        list(tqdm(executor.map(download_image, urls, [p.join(download_dir, file_name) for file_name in file_names], [dev_null] * len(urls)), total=len(urls)))
        dev_null.close()

def make_fp17k():
    print('--- FP17K ---')

    current_dir = p.dirname(p.realpath(__file__))
    fp17k_dir = current_dir + '/downloaded_data/fp17k'
    print(' Downloading images...')
    download_fp17k_images()

    metadata_df = pd.read_csv(fp17k_dir + '/fitzpatrick17k.csv')
    metadata_df = metadata_df[metadata_df['url'].notna()]

    labels_df = pd.DataFrame()

    centaur = metadata_df['fitzpatrick_centaur'].to_numpy()
    scale = metadata_df['fitzpatrick_scale'].to_numpy()
    # Use centaur if available, otherwise use scale
    labels_df['label'] = np.select([centaur > -1, scale > -1], [centaur, scale], default=-1)
    labels_df['file_name'] = metadata_df['md5hash']
    labels_df = labels_df[labels_df['label'] > -1]
    labels_df['label'] = labels_df['label'].astype(int)
    os.makedirs('fp17k', exist_ok=True)

    image_download_dir = current_dir + '/downloaded_data/fp17k/images/*'
    images = glob(image_download_dir)
    images.sort()

    copy_train_valid_test_images(images, [0.8, 0.1, 0.1], current_dir + '/fp17k')

    print(' Removing corrupted images from labels...')
    copied_images = glob(current_dir + '/fp17k/**/*/*.jpg', recursive=True)
    copied_images = [p.basename(i).replace('.jpg', '') for i in copied_images]
    labels_df = labels_df[labels_df['file_name'].isin(copied_images)]
    labels_df.to_csv(current_dir + '/fp17k/labels.csv', index=False)

def make_diverse():
    print('--- DIVERSE ---')

    # The DDI dataset has three classes, 12, 34, and 56. 
    # These correspond to the Fitzpatrick skin types 1 and 2, 3 and 4, and 5 and 6, respectively.
    # We will save the class labels as they are, and then handle the different labeling between different datasets
    # in the dataset class.
    # Also, there are no missing values in this dataset.

    current_dir = p.dirname(p.realpath(__file__))
    images_folder = current_dir + '/downloaded_data/diverse/ddidiversedermatologyimages/*.png'
    images = glob(images_folder, recursive=True)
    images.sort()

    dataset_metadata = pd.read_csv(current_dir + '/downloaded_data/diverse/ddidiversedermatologyimages/ddi_metadata.csv')
    labels_df = pd.DataFrame()
    labels_df['file_name'] = dataset_metadata['DDI_file'].str.replace('.png', '')
    labels_df['label'] = dataset_metadata['skin_tone'].astype(int)

    save_dir = current_dir + '/diverse'
    os.makedirs(save_dir, exist_ok=True)
    labels_df.to_csv(p.join(save_dir, 'labels.csv'), index=False)

    splits = [0.8, 0.1, 0.1]
    copy_train_valid_test_images(images, splits, save_dir)

if __name__ == '__main__':
    #make_pad_ufes()
    #make_fp17k()
    make_diverse()