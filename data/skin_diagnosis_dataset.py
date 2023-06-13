import os.path as p
from typing import List, Tuple, Dict, Callable, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from data.skin_color_dataset import SkinColorDetectionDataset

class SkinDiagnosisDataset(SkinColorDetectionDataset):
  def __init__(self, 
               subset: Literal['train', 'valid', 'test', 'all'], 
               dataset_folder: str, 
               subjects: Optional[List[str]] = None, 
               augment = False,
               colorspace: Literal['lab', 'rgb']='lab',
               label_encoding: str = 'class'):
    # ignore label_encoding
    super().__init__(subset, dataset_folder, subjects, augment, colorspace, label_encoding='class')

    class_names = self.labels_df['label_diag'].unique()
    class_names.sort()
    self.all_classes = list(class_names)
        
  def __len__(self):
    return len(self.labels_df)
  
  def get_class_tensor(self, index: int) -> torch.Tensor:
    label = self.labels_df.iloc[index]['label_diag']
    class_idx = self.all_classes.index(label)
    return torch.tensor(class_idx, dtype=torch.long)
  
def FP17KDiagnosisDataset(**kwargs) -> SkinColorDetectionDataset:
  return SkinDiagnosisDataset(dataset_folder='fp17k', **kwargs)
