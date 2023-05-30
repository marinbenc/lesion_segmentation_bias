import random
import os
import os.path as p
import shutil
import json
import sys
from typing import List, Tuple, Dict, Callable, Optional, Literal

import fire
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from monai.losses.dice import DiceLoss

import data.datasets as data
from data.stratified_sampler import StratifiedSampler
import utils
import models.models as models
from trainer import Trainer

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

ModelType = Literal['lesion_seg', 'skin_detection']
DatasetType = Literal['seg_isic', 'skin_isic']

def get_model(
    model_type: ModelType, 
    log_dir: str, 
    dataset: DatasetType, 
    device: str, 
    fold: int) -> nn.Module:
    if model_type == 'lesion_seg':
        model = models.get_segmentation_model(dataset, device)
    elif model_type == 'skin_detection':
        model = models.get_detection_model(dataset, device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model

def train(*, 
    model_type: Literal['lesion_seg', 'skin_detection'],
    batch_size: int = 64, 
    epochs: int = 100, 
    lr: float = 0.001, 
    dataset: DatasetType,
    colorspace: Literal['rgb', 'lab', 'white', 'dist'] = 'lab',
    augment_skin_color: bool = False,
    skin_color_detection_method: Literal['knn', 'nn'] = 'knn',
    log_name: str, 
    device: str = 'cuda',
    folds: int = 1,
    stratified_sampling: bool = False,
    stratified_sample_skin_color_augmentation: bool = False,
    overwrite: bool = False,
    workers: int = 8):
    """
    Train a detection or segmentation model. 
    
    The model checkpoints are saved in runs/log-name/model-type. 

    Args:
        model_type ('seg' or 'skin_detection'): The type of model to train.
        dataset ('seg_isic' or 'skin_isic'): The dataset to train on.
        folds (int): The number of folds to train on. If folds > 1, then the model is trained on a k-fold cross validation.
        stratified_sampling (bool): If True, then the training and validation sets are sampled such that the proportion of skin images is the same in both sets.
        augment_skin_color (bool): If True, will tint images during training.
        stratified_sample_skin_color_augmentation (bool): If True, will use stratified sampling to augment skin colors.
        colorspace ('lab', 'rgb', 'white' or 'dist'): The colorspace to use for the model. See paper for details on 'white' and 'dist'.
        skin_color_detection_method ('knn' or 'nn'): Which CSV file to use for the skin colors. See data/segmentation_dataset.py for details.
        log_name (str): The name of the log directory. The model checkpoints are saved in runs/log-name/model-type.
        device (str): The device to train on.
        overwrite (bool): If True, the log_name directory is deleted before training.
    """
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)
    os.makedirs(name=f'runs/{log_name}', exist_ok=True)

    datasets = []

    dataset_args_valid = {
        'subset': 'valid',
        'augment': False,
        # If using skin-color-tinted images, then the validation should be on color images.
        # Otherwise if 'white', then the validation should be on white-balanced images.
        'colorspace': 'rgb' if colorspace == 'white' and augment_skin_color else colorspace,
        'augment_skin_color': False,
        'stratified_sample_skin_color_augmentation': stratified_sample_skin_color_augmentation,
        'skin_color_detection_method': skin_color_detection_method
    }

    dataset_args_train = {
        **dataset_args_valid,
        'subset': 'train',
        'augment': True,
        'augment_skin_color': augment_skin_color,
    }
    
    dataset_class = data.get_dataset_class(dataset)

    if folds == 1:
        train_dataset = dataset_class(**dataset_args_train)
        valid_dataset = dataset_class(**dataset_args_valid)
        datasets.append((train_dataset, valid_dataset))
        json_dict = {
            'train_subjects': list(train_dataset.subjects),
            'valid_subjects': list(valid_dataset.subjects)
        }
        with open(f'runs/{log_name}/subjects.json', 'w') as f:
            json.dump(json_dict, f)
    else:
        whole_args_valid = dataset_args_valid.copy()
        whole_args_valid['subset'] = 'all'
        whole_args_train = dataset_args_train.copy()
        whole_args_train['subset'] = 'all'
        whole_dataset = dataset_class(**whole_args_valid)
        subject_ids = list(whole_dataset.subjects)
        subject_ids = sorted(subject_ids)

        existing_split = p.join('runs', log_name, 'subjects.json')
        if p.exists(existing_split):
            print('Using existing subject split')
            with open(existing_split, 'r') as f:
                json_dict = json.load(f)
                splits = zip(json_dict['train_subjects'], json_dict['valid_subjects'])
        else:
            kfold = KFold(n_splits=folds, shuffle=True, random_state=2022)
            splits = list(kfold.split(subject_ids))
            # convert from indices to subject ids
            splits = [([subject_ids[idx] for idx in train_idx], [subject_ids[idx] for idx in valid_idx]) for train_idx, valid_idx in splits]
            json_dict = {
                'train_subjects': [ids for (ids, _) in splits],
                'valid_subjects': [ids for (_, ids) in splits]
            }
            with open(f'runs/{log_name}/subjects.json', 'w') as f:
                json.dump(json_dict, f)

        for fold, (train_ids, valid_ids) in enumerate(splits):
            train_dataset = dataset_class(**whole_args_train, subjects=train_ids)
            valid_dataset = dataset_class(**whole_args_valid, subjects=valid_ids)
            # check for data leakage
            intersection = set(train_dataset.file_names).intersection(set(valid_dataset.file_names))
            assert len(intersection) == 0, f'Found {len(intersection)} overlapping files in fold {fold}'

    os.makedirs(name=f'runs/{log_name}/{model_type}', exist_ok=True)
    if command_string and command_string is not None:
        command_file = p.join('runs', log_name, model_type, 'command.sh')
        with open(command_file, 'w') as f:
            f.write(command_string)

    for fold, (train_dataset, valid_dataset) in enumerate(datasets):
        print('----------------------------------------')
        print(f'Fold {fold}')
        print('----------------------------------------')

        log_dir = f'runs/{log_name}/{model_type}/fold{fold}'
        if p.exists(log_dir):
            if overwrite:
                shutil.rmtree(log_dir)
            else:
                raise ValueError(f'Log directory already exists: {log_dir}. Use --overwrite to overwrite.')

        if stratified_sampling:
            train_sampler = StratifiedSampler(train_dataset)
            valid_sampler = StratifiedSampler(valid_dataset)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init, num_workers=workers, sampler=train_sampler)
            valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers, sampler=valid_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init, num_workers=workers)
            valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers)

        if model_type == 'lesion_seg':
            dice_loss = DiceLoss()
            def loss(pred, target):
                target_seg = target['seg']
                return dice_loss(pred, target_seg)
        elif model_type == 'skin_detection':
            loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        model = get_model(model_type, log_dir, train_dataset, device, fold)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True, min_lr=1e-15, eps=1e-15)
        
        trainer = Trainer(model, optimizer, loss, train_loader, valid_loader, 
                          log_dir=log_dir, checkpoint_name=f'{model_type}_best_fold={fold}.pth', scheduler=scheduler)
        trainer.train(epochs)

if __name__ == '__main__':
    command_string = ' '.join(sys.argv)
    fire.Fire(train)