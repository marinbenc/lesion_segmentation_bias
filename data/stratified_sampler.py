import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch

from data.skin_color_dataset import SkinColorDetectionDataset
from data.skin_diagnosis_dataset import SkinDiagnosisDataset

def StratifiedSampler(dataset: SkinColorDetectionDataset):
  """
  Returns a stratified sampler for the dataset. The weights are based on the distribution of the ita_angle attribute in the dataset.
  """
  classes = dataset.all_classes
  labels_key = 'label_diag' if isinstance(dataset, SkinDiagnosisDataset) else 'label'
  labels = dataset.labels_df[labels_key].to_numpy()

  class_counts = np.zeros(len(classes))
  for i, class_name in enumerate(classes):
    class_counts[i] = np.sum(labels == class_name)

  weights = 1 / class_counts
  weights = weights / np.sum(weights)

  sample_weights = np.zeros(len(labels))
  for i, label in enumerate(labels):
    sample_weights[i] = weights[classes.index(label)]

  print(weights)
  sampler = WeightedRandomSampler(list(sample_weights), len(sample_weights) * 2, replacement=True)
  return sampler
