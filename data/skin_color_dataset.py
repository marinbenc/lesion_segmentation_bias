import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

import os

class SkinColorDetectionDataset(torch.utils.data.Dataset):
  f"""
  A dataset for detecting skin color from dermatology images.

  There are three possible classes: 12, 34 and 56. These correspond to Fitzpatrick skin types 1-2, 3-4 and 5-6 respectively.

  The labels.csv file must have the following columns:
    file_name (str): The name of the image file, without the extension.
    label (int): The class label of the image.

    There are two possible formats for the label. Either it can be a single digit integer [1, 6] corresponding to the Fitzpatrick skin type, 
    or it can be a two digit number in {12, 34, 56} corresponding to the skin type classes. The former will be converted to the latter in the dataset class.

  Args:
    subset ('train', 'valid', 'test' or 'all'): The subset of the dataset to use. Can be 'train', 'valid', 'test', or 'all'.
    dataset_folder (str): The folder containing the images and labels. The directory must be in the same directory as this file. 
    Expected folder structure:
      dataset_folder
      ├── labels.csv (dominant colors for each image based on k-nearest neighbors)
      ├── train
      │   ├── <subject_name 0>.jpg
      │   ├── <subject_name 1>.jpg
      │   └── ...
      ├── valid
      │   ├── <subject_name 0>.jpg
      │   ├── <subject_name 1>.jpg
      │   └── ...
      ├── test
      │   ├── <subject_name 0>.jpg
      │   ├── <subject_name 1>.jpg
      │   └── ...
    subjects (List[str]): 
      A list of subject names to use. If None, all subjects are used.
    augment (bool): 
      Whether to augment the dataset. If colorspace is 'dist', the image will be tinted with a randomly sampled color.
    colorspace ('lab' or 'rgb'): 
      The colorspace to use.
    classes (List[int]): 
      A list of classes (12, 34, 56) to use. If None, all classes are used. Useful for training only on a subset of the classes.

    Attributes:
      subjects (set[str]): Names of the subjects in the dataset.
      all_classes ([int]): List of class labels == [12, 34, 56].
      labels_df (pd.DataFrame): The labels dataframe.
  """
  def __init__(self, 
               subset: Literal['train', 'valid', 'test', 'all'], 
               dataset_folder: str, 
               subjects: Optional[List[str]] = None, 
               augment = False, 
               colorspace: Literal['lab', 'rgb']='lab', 
               classes: List[int] = [12, 34, 56]):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.all_classes = [12, 34, 56]
    self.augment = augment

    assert self.colorspace in ['lab', 'rgb']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    file_paths = np.array(self._get_files(directories))
    file_subjects = [self._get_subject_from_file_name(f) for f in file_paths]
    
    self.labels_df = pd.read_csv(p.join(self.dataset_folder, 'labels.csv'))
    # Keep only the subjects that are in directories
    self.labels_df = self.labels_df[self.labels_df['file_name'].isin(file_subjects)]

    # Append the file paths to the labels dataframe
    df_subject_names = self.labels_df['file_name'].as_numpy()
    self.labels_df['file_path'] = file_paths[np.argsort(df_subject_names)]

    if subjects is not None:
        # TODO: Check if this is working
        self.labels_df = self.labels_df[self.labels_df['file_name'].isin(subjects)]
    self.labels_df.sort_values(by=['file_name'], inplace=True)

    # TODO: classes is not used here
    
  def _get_files(self, directories):
    file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      directory_files = os.listdir(directory)
      directory_files = [p.join(directory, f) for f in directory_files]
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
    return len(self.labels_df)
  
  def get_item_np(self, idx, augmentation=None):
    current_file = self.labels_df.iloc[idx]['file_path']
    input = cv.imread(current_file)

    if self.colorspace == 'lab':
      input = cv.cvtColor(input, cv.COLOR_BGR2LAB)

    elif self.colorspace == 'rgb':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
        
    input = input.transpose(2, 0, 1)
    
    if augmentation is not None:
      input = input.transpose(1, 2, 0)
      transformed = augmentation(image=input)
      input = transformed['image']
      input = input.transpose(2, 0, 1)

    return input

  def __getitem__(self, idx):
    input = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    to_tensor = ToTensorV2()
    input = input.astype(np.float32)
    input = input / 255.
    
    input_tensor = to_tensor(image=input.transpose(1, 2, 0)).values()
    input_tensor = input_tensor.float()
    label_tensor = label_tensor.unsqueeze(0).float()

    # one-hot encoding
    class_label = self.labels_df.iloc[idx]['label']
    class_index = self.all_classes.index(class_label)
    class_label_tensor = torch.zeros(len(self.all_classes))
    class_label_tensor[class_index] = 1

    #plt.imshow(input.transpose(1, 2, 0))
    #plt.show()
    #plt.imshow(input.transpose(1, 2, 0)[:,:,[3,4,5]])
    #plt.show()

    return input_tensor, class_label_tensor
  
def FP17KDataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='fp17k', **kwargs)

def DiverseDataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='diverse', **kwargs)

def PADUFES20Dataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='pad_ufes_20', **kwargs)