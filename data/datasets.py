from data.segmentation_dataset import LesionSegDatasetDermis, LesionSegDatasetDermQuest, LesionSegDatasetISIC
from data.skin_color_dataset import DiverseDataset, FP17KDataset

dataset_to_class = {
    'seg_isic': LesionSegDatasetISIC,
    'seg_dermis': LesionSegDatasetDermis,
    'seg_dermquest': LesionSegDatasetDermQuest,
    'det_diverse': DiverseDataset,
    'det_fp17k': FP17KDataset,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name):
  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]