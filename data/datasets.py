from typing import List, Union
from functools import partial
import pandas as pd
import numpy as np

from data.segmentation_dataset import (
    LesionSegDatasetDermis, LesionSegDatasetDermQuest, LesionSegDatasetISIC, 
    LesionSegmentationDataset, LesionSegDatasetDermofit, LesionSegDatasetPH2)
from data.skin_color_dataset import DiverseDataset, FP17KDataset, PADUFES20Dataset, SkinColorDetectionDataset
from data.skin_diagnosis_dataset import FP17KDiagnosisDataset

dataset_to_class = {
    'seg_isic': LesionSegDatasetISIC,
    'seg_dermis': LesionSegDatasetDermis,
    'seg_dermquest': LesionSegDatasetDermQuest,
    'seg_ph2': LesionSegDatasetPH2,
    'seg_dermofit': LesionSegDatasetDermofit,
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
    if isinstance(dataset1, SkinColorDetectionDataset):
      dataset1.labels_df = pd.concat([dataset1.labels_df, dataset2.labels_df])
      dataset1.subjects = np.concatenate([dataset1.subjects, dataset2.subjects])
    elif isinstance(dataset1, LesionSegmentationDataset):
      dataset1.skin_colors_df = pd.concat([dataset1.skin_colors_df, dataset2.skin_colors_df])
      dataset1.subjects = set(list(dataset1.subjects) + list(dataset2.subjects))
      dataset1.skin_colors = dataset1.skin_colors + dataset2.skin_colors
      dataset1.file_names = dataset1.file_names + dataset2.file_names
      dataset1.subject_id_for_idx += dataset2.subject_id_for_idx
  return dataset1