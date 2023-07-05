import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

from torchvision.models import ResNet18_Weights

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
    label_encoding ('class', 'ordinal-2d' or 'ordinal-1d'):
      The label encoding to use. 'class' is [0, 1, 2], 'ordinal-2d' is [[1,0,0], [1,1,0], [1,1,1]], and 'ordinal-1d' is [1/6, 3/6, 5/6]
    prediction (bool):
      Whether to use the dataset for prediction. If True, the labels are not loaded.

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
               label_encoding: Literal['class', 'ordinal-2d', 'ordinal-1d'] = 'ordinal-1d',
               prediction: bool = False):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.all_classes = [12, 34, 56]
    self.augment = augment
    self.model_transforms = ResNet18_Weights.DEFAULT.transforms()
    self.label_encoding = label_encoding
    self.prediction = prediction

    assert self.colorspace in ['lab', 'rgb']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    file_paths = np.array(self._get_files(directories))
    file_subjects = [self._get_subject_from_file_name(f) for f in file_paths]
    
    if prediction:
      self.labels_df = pd.DataFrame({'file_name': file_subjects, 'file_path': file_paths})
    else:
      this_file_dir = p.dirname(__file__)
      self.labels_df = pd.read_csv(
        p.join(this_file_dir, self.dataset_folder, 'labels.csv'), 
        dtype={'label': int, 'file_name': str})
      # Keep only the subjects that are in directories
      self.labels_df = self.labels_df[self.labels_df['file_name'].isin(file_subjects)]

      # Append the file paths to the labels dataframe
      df_subject_names = self.labels_df['file_name'].to_numpy()
      for i, subject_name in enumerate(df_subject_names):
          file_path = file_paths[file_subjects.index(subject_name)]
          self.labels_df.at[i, 'file_path'] = file_path

      labels_remapping = {
          1: 12,
          2: 12,
          3: 34,
          4: 34,
          5: 56,
          6: 56
      }

      # Remap the labels to the classes
      if self.labels_df.iloc[0]['label'] in [1, 2, 3, 4, 5, 6]:
          self.labels_df['label'] = self.labels_df['label'].map(labels_remapping)

    if subjects is not None:
        # TODO: Check if this is working
        self.labels_df = self.labels_df[self.labels_df['file_name'].isin(subjects)]
    self.labels_df.sort_values(by=['file_name'], inplace=True)

    self.subjects = self.labels_df['file_name'].to_numpy()
    self.subject_id_for_idx = self.subjects
    self.labels_df = self.labels_df.dropna()
    
  def _get_files(self, directories):
    file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      if self.prediction and p.exists(p.join(directory, 'input')):
        directory = p.join(directory, 'input')
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
      A.RandomGamma(p=0.7, gamma_limit=(80, 180)),
      A.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
      A.Flip(p=0.5),
      A.ShiftScaleRotate(p=0.4, rotate_limit=10, scale_limit=0.2, shift_limit=0.2, border_mode=cv.BORDER_REFLECT101),
      #A.GridDistortion(p=0.4, border_mode=cv.BORDER_CONSTANT, value=0)
    ])
  
  def __len__(self):
    #return 64
    return len(self.labels_df)
  
  def get_item_np(self, idx, augmentation=None):
    current_file = self.labels_df.iloc[idx]['file_path']
    input = cv.imread(current_file)

    # color quantization
    # input_subsampled = input[::2, ::2, :]
    # input_subsampled = input_subsampled.reshape((-1, 3))
    # from sklearn.cluster import KMeans
    # clt = KMeans(n_clusters=5, random_state=42, n_init='auto')
    # clt.fit(input_subsampled)
    # quantized = clt.cluster_centers_[clt.labels_]
    # quantized = quantized.reshape(input_subsampled.shape)
    # quantized = quantized.reshape((input.shape[0] // 2, input.shape[1] // 2, 3))
    # quantized = quantized.astype(np.uint8)
    # input = quantized
    #plt.imshow(quantized)
    #plt.show()

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

#   def label_to_class(self, label):
#     class_vector = np.zeros(len(self.all_classes))
#     # Ordinal encoding of the classes
#     if label == 12:
#         class_vector[0] = 1
#     elif label == 34:
#         class_vector[0, 1] = 1
#     elif label == 56:
#         class_vector[0, 1, 2] = 1
#     return class_vector

  def get_class_tensor(self, idx: int) -> torch.Tensor:
    class_idx = self.labels_df.iloc[idx]['label']
    class_idx = self.all_classes.index(class_idx)
    class_idx = torch.tensor(class_idx).long()

    if self.label_encoding == 'ordinal-2d':
        class_tensor = torch.zeros(len(self.all_classes)).float()
        class_tensor[:class_idx+1] = 1
    elif self.label_encoding == 'ordinal-1d':
        if class_idx == 0:
            class_tensor = torch.tensor(1/6.).float()
        elif class_idx == 1:
            class_tensor = torch.tensor(3/6.).float()
        elif class_idx == 2:
            class_tensor = torch.tensor(5/6.).float()
        else:
            raise Exception('Invalid class index')
        class_tensor = class_tensor.unsqueeze(0)
    elif self.label_encoding == 'class':
        class_tensor = class_idx.long()
    else:
        raise Exception('Invalid label encoding')
    return class_tensor

  def __getitem__(self, idx):
    input = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    input = input.astype(np.float32)
    
    #input_tensor = to_tensor(image=input.transpose(1, 2, 0))['image']
    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.float()

    # # # skin color thresholding
    # # 0.0 <= (r-g)/(r+g) <= 0.5
    # rg_ratio = (input_tensor[0] - input_tensor[1]) / (input_tensor[0] + input_tensor[1] + 1e-6)
    # # b / (r+g) <= 0.5
    # b_ratio = input_tensor[2] / (input_tensor[0] + input_tensor[1] + 1e-6)
    # skin_mask = (rg_ratio >= 0.0) & (rg_ratio <= 0.5) & (b_ratio <= 0.5)
    # input_tensor[0][~skin_mask] = 0.0
    # input_tensor[1][~skin_mask] = 0.0
    # input_tensor[2][~skin_mask] = 0.0

    # imagenet normalization
    input_tensor = input_tensor / 255.0
    # normalize to imagenet mean and std
    input_tensor[0] = (input_tensor[0] - 0.485) / 0.229
    input_tensor[1] = (input_tensor[1] - 0.456) / 0.224
    input_tensor[2] = (input_tensor[2] - 0.406) / 0.225
    # crop to 224x224
    if self.prediction:
      input_tensor = torch.functional.F.interpolate(input_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
    else:
      input_tensor = input_tensor[:, 16:240, 16:240]

    if self.prediction:
      return input_tensor
    
    class_tensor = self.get_class_tensor(idx)    
    # plt.imshow(input.transpose(1, 2, 0) / 255.0)
    # file_name = self.labels_df.iloc[idx]['file_name']
    # plt.title(f"{file_name} - {self.labels_df.iloc[idx]['label']}")
    # plt.show()

    return input_tensor, class_tensor
  
def FP17KDataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='fp17k', **kwargs)

def DiverseDataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='diverse', **kwargs)

def PADUFES20Dataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinColorDetectionDataset(dataset_folder='pad_ufes', **kwargs)