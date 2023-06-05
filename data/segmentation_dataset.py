import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal

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

class LesionSegmentationDataset(torch.utils.data.Dataset):
  """
  A dataset for segmenting skin lesions.

  Args:
    subset ('train', 'valid', 'test' or 'all'): The subset of the dataset to use. Can be 'train', 'valid', 'test', or 'all'.
    dataset_folder (str): The folder containing the images and labels. The directory must be in the same directory as this file. 
    Expected folder structure:
      dataset_folder
      ├── skin_colors_knn.csv (dominant colors for each image based on k-nearest neighbors)
      ├── skin_colors_nn.csv (dominant colors for each image based on neural network)
      ├── skin_colors_cc.csv (dominant colors for each image based on color constancy, see included examples for format)
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
      ita_angles (List[float]): The ITA angle for each image in the dataset.
      file_names (List[str]): The file name for each image in the dataset.
      classes (List[int]): The class for each image in the dataset.
      skin_colors (List[[int, int, int]]): The RGB skin color for each image in the dataset.
  """
  def __init__(self, 
               subset: Literal['train', 'valid', 'test', 'all'], 
               dataset_folder: str, 
               subjects: Optional[List[str]] = None, 
               augment = False, 
               colorspace: Literal['lab', 'rgb', 'dist', 'white']='lab', 
               classes: Optional[List[int]] = None,
               augment_skin_color: bool = False,
               stratified_sample_skin_color_augmentation: bool = False,
               skin_color_detection_method: Literal['knn', 'nn', 'cc']='knn'):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.num_classes = 6
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

    file_dir = os.path.dirname(os.path.realpath(__file__))
    self.skin_colors_df = pd.read_csv(p.join(file_dir, 'dominant_colors.csv')) # TODO: Make this dataset-dependent
    self.skin_colors_df['image'] = self.skin_colors_df['image'].str.replace('.jpg', '')
    self.skin_colors_df.set_index('image', inplace=True)
    self.skin_colors = [self.skin_colors_df.loc[s][['R', 'G', 'B']].astype(np.uint8).values for s in self.subject_id_for_idx]

    self.ita_angles = [pre.get_ita_angle(c) for c in self.skin_colors]
    self.classes = [self.class_for_ita_angle(ita) for ita in self.ita_angles]

    if classes is not None:
      new_idxs = [idx for idx, c in enumerate(self.classes) if c in classes]
      self.file_names = [self.file_names[idx] for idx in new_idxs]
      self.subject_id_for_idx = [self.subject_id_for_idx[idx] for idx in new_idxs]
      self.skin_colors = [self.skin_colors[idx] for idx in new_idxs]
      self.ita_angles = [self.ita_angles[idx] for idx in new_idxs]
      self.classes = [self.classes[idx] for idx in new_idxs]
  
  def class_for_ita_angle(self, ita_angle):
    """
    Returns the class label for the given ITA angle based on standard ITA ranges:
      <-30: 0 - Dark
      -30 to 10: 1 - Brown
      10 to 28: 2 - Tan
      28 to 41: 3 - Intermediate
      41 to 55: 4 - Light
      >55: 5 - Very light
    """
    # TODO: Refactor this to match Fitzpatrick-17 dataset
    range_limits = [-30, 10, 28, 41, 55]
    for i in range(len(range_limits)):
      if ita_angle < range_limits[i]:
        return i
    return len(range_limits)
  
  def random_skin_color(self, stratified=False):
    def b_center(ita):
      # b_center is 0 for ita = +- 90 and 20 for ita = 50
      return 20 * np.cos(ita * np.pi / 180)

    # TODO: Refactor this to match Fitzpatrick-17 dataset
    ita_range_for_class = [
      (-90, -30),
      (-30, 10),
      (10, 28),
      (28, 41),
      (41, 55),
      (55, 90)
    ]

    if stratified:
      class_probs = np.zeros(self.num_classes)
      for c in range(self.num_classes):
        sum = np.sum(self.classes == c)
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
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
      skin_color = self.skin_colors[idx]
      if augmentation is not None:
        aug = np.random.normal(0, 10, 3)
        aug = np.round(aug).astype(int)
        skin_color = skin_color.astype(int) + aug
        skin_color = skin_color.astype(np.uint8)
      input = pre.dist_img(input, skin_color)
    
    elif self.colorspace == 'white':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
      input, _, _ = pre.shade_of_gray_cc(input, power=6, gamma=1.2)
      if self.augment_skin_color:
        color = self.random_skin_color(stratified=self.stratified_sample_skin_color_augmentation)
        illuminant = pre.illuminant_from_color(color)
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

    class_label = self.classes[idx]
    class_label_tensor = torch.zeros(self.num_classes)
    class_label_tensor[class_label] = 1

    #plt.imshow(input.transpose(1, 2, 0))
    #plt.show()
    #plt.imshow(input.transpose(1, 2, 0)[:,:,[3,4,5]])
    #plt.show()

    return input_tensor, {'seg': label_tensor, 'aux': class_label_tensor}