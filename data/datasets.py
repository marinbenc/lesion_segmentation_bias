from typing import List, Union
from functools import partial
import pandas as pd
import numpy as np

from data.segmentation_dataset import LesionSegDatasetDermis, LesionSegDatasetDermQuest, LesionSegDatasetISIC
from data.skin_color_dataset import DiverseDataset, FP17KDataset, PADUFES20Dataset
from data.skin_diagnosis_dataset import FP17KDiagnosisDataset

dataset_to_class = {
    'seg_isic': LesionSegDatasetISIC,
    'seg_dermis': LesionSegDatasetDermis,
    'seg_dermquest': LesionSegDatasetDermQuest,
    'det_diverse': DiverseDataset,
    'det_fp17k': FP17KDataset,
    'det_pad_ufes_20': PADUFES20Dataset,
    'diag_fp17k': FP17KDiagnosisDataset,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name: Union[str, List[str]]):
  print(dataset_name)
  if isinstance(dataset_name, list):
    return partial(composed_dataset, dataset_names=dataset_name)

  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]

def composed_dataset(dataset_names: List[str], **kwargs):
  datasets = [get_dataset_class(name)(**kwargs) for name in dataset_names]
  dataset1 = datasets[0]
  for dataset2 in datasets[1:]:
    dataset1.labels_df = pd.concat([dataset1.labels_df, dataset2.labels_df])
    dataset1.subjects = np.concatenate([dataset1.subjects, dataset2.subjects])
  
  return dataset1