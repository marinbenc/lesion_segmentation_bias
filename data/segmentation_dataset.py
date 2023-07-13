import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from skimage import color

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

import preprocessing_utils as pre

import os

def LesionSegDatasetISIC(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='isic', **kwargs)

def LesionSegDatasetDermQuest(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='dermquest', **kwargs)

def LesionSegDatasetDermis(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='dermis', **kwargs)

def LesionSegDatasetPH2(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='ph2', **kwargs)

def LesionSegDatasetDermofit(subset, **kwargs):
  return LesionSegmentationDataset(subset=subset, dataset_folder='dermofit', **kwargs)

class LesionSegmentationDataset(torch.utils.data.Dataset):
  """
  A dataset for segmenting skin lesions.

  Args:
    subset ('train', 'valid', 'test' or 'all'): The subset of the dataset to use. Can be 'train', 'valid', 'test', or 'all'.
    dataset_folder (str): The folder containing the images and labels. The directory must be in the same directory as this file. 
    Expected folder structure:
      dataset_folder
      ├── skin_color_prediction.csv (Fitzpatrick skin type for each image, see included examples for format)
      ├── train
      │   ├── input
      │   │   ├── <subject_name 0>.png
      │   │   ├── <subject_name 1>.png
      │   │   └── ...
      │   └── label
      │       ├── <subject_name 0>.png
      │       ├── <subject_name 1>.png
      │       └── ...
      ├── valid
      │   ├── input
      │   │   ├── ...
      │   └── label
      │       ├── ...
      ├── test
      │   ├── input
      │   │   ├── ...
      │   └── label
      │       ├── ...
    subjects (List[str]): 
      A list of subject names to use. If None, all subjects are used.
    augment (bool): 
      Whether to augment the dataset. If colorspace is 'dist', the image will be tinted with a randomly sampled color.
    colorspace ('lab', 'rgb', 'white' or 'dist'): 
      The colorspace to use. Please see the paper for more information on 'white' and 'dist'.
    classes (List[int]): 
      A list of classes in [0, 5] to use, ordered by increasing ITA angle. If None, all classes are used. Useful for training only on a subset of the classes.
    augment_skin_color (bool):
      if True, images will be tinted randomly. This is only used if colorspace='dist' or 'white'.
    stratified_sample_skin_color_augmentation (bool): 
      if True, will use the class histogram to sample less common skin colors more often. Otherwise will sample uniformly.
      This is only used if augment=True and colorspace='dist'.
    skin_color_detection_method ('knn', 'nn' or 'cc'):
      Which skin color CSV to use. Has to be included in the dataset folder as skin_colors_<method>.csv.

    Attributes:
      subjects (set[str]): Names of the subjects in the dataset.
      num_classes (int): Number of classes in the dataset.
      subject_id_for_idx (List[str]): The subject name for each image in the dataset.
      file_names (List[str]): The file name for each image in the dataset.
      skin_colors (List[int]): The Fitzpatrick type label of each image (12, 34 or 56).
  """
  def __init__(self, 
               subset: Literal['train', 'valid', 'test', 'all'], 
               dataset_folder: str, 
               subjects: Optional[List[str]] = None, 
               augment = False, 
               colorspace: Literal['lab', 'rgb', 'dist', 'white']='lab', 
               classes: Optional[List[int]] = None,
               augment_skin_color: bool = False,
               stratified_sample_skin_color_augmentation: bool = False):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.num_classes = 3
    self.augment = augment
    self.augment_skin_color = augment_skin_color
    self.stratified_sample_skin_color_augmentation = stratified_sample_skin_color_augmentation

    # TODO: Switch to using skin color classes instead of actual colors?

    assert self.colorspace in ['lab', 'rgb', 'dist', 'white']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    self.file_names = self._get_files(directories)
    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)

    file_dir = Path(p.dirname(__file__))
    skin_color_csv_file = f'skin_color_prediction.csv'
    self.skin_colors_df = pd.read_csv(file_dir / self.dataset_folder / skin_color_csv_file, dtype={'file_name': str, 'nn_skin_type': int})
    # TODO: Check the following line
    self.skin_colors_df['file_name'] = self.skin_colors_df['file_name'].str.replace('.jpg', '')
    self.skin_colors_df.set_index('file_name', inplace=True)
    self.skin_colors = [self.skin_colors_df.loc[s]['nn_skin_type'] for s in self.subject_id_for_idx]

    if classes is not None:
      new_idxs = [idx for idx, c in enumerate(self.skin_colors) if c in classes]
      self.file_names = [self.file_names[idx] for idx in new_idxs]
      self.subject_id_for_idx = [self.subject_id_for_idx[idx] for idx in new_idxs]
      self.skin_colors = [self.skin_colors[idx] for idx in new_idxs]
    
  def random_skin_color(self, stratified=False):
    def b_center(ita):
      # b_center is 0 for ita = +- 90 and 20 for ita = 50
      return 20 * np.cos(ita * np.pi / 180)

    # Based on Kinyanjui el al. Estimating Skin Tone and Effects
    # on Classification Performance in Dermatology Datasets.
    # https://arxiv.org/abs/1910.13268
    ita_range_for_class = [
      (41, 90),   # 12
      (19, 41),   # 34
      (-90, 19)   # 56
    ]

    if stratified:
      class_probs = np.zeros(self.num_classes)
      class_labels = [12, 34, 56]
      for c in range(self.num_classes):
        sum = np.sum(self.skin_colors == class_labels[c])
        if sum == 0:
          class_probs[c] = 1
        else:
          class_probs[c] = 1 / sum
      
      class_probs /= np.sum(class_probs)
    else:
      class_probs = np.ones(self.num_classes) / self.num_classes

    class_idx = np.random.choice(self.num_classes, p=class_probs)

    random_ita = np.random.uniform(*ita_range_for_class[class_idx])
    random_b = np.random.normal(b_center(random_ita), 6)
    random_l = random_b * np.tan(random_ita * np.pi / 180) + 50
    random_a = np.random.normal(10, 2)

    random_lab = np.array([random_l, random_a, random_b])

    random_rgb = color.lab2rgb(random_lab.reshape(1, 1, 3)).reshape(3)
    return random_rgb

  def _get_files(self, directories):
    file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      directory_files = os.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      file_names += directory_files
      file_names.sort()
    return file_names

  def _get_subject_from_file_name(self, file_name):
    return file_name.split('/')[-1].split('.')[0]
  
  def get_train_augmentation(self):
    return A.Compose([
      #A.RandomGamma(p=0.7, gamma_limit=(80, 180)),
      #A.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
      A.Flip(p=0.4),
      A.ShiftScaleRotate(p=0.4, rotate_limit=90, scale_limit=0.1, shift_limit=0.1, border_mode=cv.BORDER_CONSTANT, value=0, rotate_method='ellipse'),
      A.GridDistortion(p=0.4, border_mode=cv.BORDER_CONSTANT, value=0)
    ])
  
  def __len__(self):
    return len(self.file_names)
  
  def get_item_np(self, idx, augmentation=None):
    current_file = self.file_names[idx]

    input = cv.imread(current_file.replace('label/', 'input/').replace('.png', '.jpg'))

    if self.colorspace == 'lab':
      input = cv.cvtColor(input, cv.COLOR_BGR2LAB)
    elif self.colorspace == 'rgb':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    elif self.colorspace == 'dist':
      # TODO: Remove dist if not used
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
      _, _, skin_color = pre.shade_of_gray_cc(input, power=6, gamma=1.2)
      if augmentation is not None:
        aug = np.random.normal(0, 10, 3)
        aug = np.round(aug).astype(int)
        skin_color = skin_color.astype(int) + aug
        skin_color = skin_color.astype(np.uint8)
      input = pre.dist_img(input, skin_color)
    elif self.colorspace == 'white':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
      # first use color constancy to remove skin color
      input, _, _ = pre.shade_of_gray_cc(input, power=6, gamma=1.2)
      if self.augment_skin_color:
        color = self.random_skin_color(stratified=self.stratified_sample_skin_color_augmentation)
        illuminant = pre.illuminant_from_color(color)
        # then tint the image given a different random skin color
        input = input / illuminant
      input = np.clip(input, 0, 255)
      input = input.astype(np.uint8)
    
    input = input.transpose(2, 0, 1)

    mask = cv.imread(current_file, cv.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    if augmentation is not None:
      input = input.transpose(1, 2, 0)
      transformed = augmentation(image=input, mask=mask)
      input = transformed['image']
      input = input.transpose(2, 0, 1)
      mask = transformed['mask']

    mask = mask.astype(np.float32)
    return input, mask

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    to_tensor = ToTensorV2()
    input = input.astype(np.float32)
    input = input / 255.
    # TODO: Check if input is already in range 0-1 for white and dist color spaces
    
    input_tensor, label_tensor = to_tensor(image=input.transpose(1, 2, 0), mask=label).values()
    input_tensor = input_tensor.float()
    label_tensor = label_tensor.unsqueeze(0).float()

    class_label = self.skin_colors[idx]
    if class_label == 12:
      class_label = 0
    elif class_label == 34:
      class_label = 1
    elif class_label == 56:
      class_label = 2
    class_label_tensor = torch.tensor(class_label).long()

    #plt.imshow(input.transpose(1, 2, 0))
    #plt.show()

    return input_tensor, {'seg': label_tensor, 'aux': class_label_tensor}
