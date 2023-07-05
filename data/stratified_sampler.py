import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch

from data.skin_color_dataset import SkinColorDetectionDataset
from data.skin_diagnosis_dataset import SkinDiagnosisDataset

def StratifiedSampler(dataset: SkinColorDetectionDataset):
  """
  Returns a stratified sampler for the dataset. The weights are based on the distribution of the ita_angle attribute in the dataset.
  """
  label_per_sample = dataset.skin_colors
  classes = np.unique(label_per_sample)
  classes.sort()

  class_to_idx = {classes[i]: i for i in range(len(classes))}

  class_sample_count = np.array([len(np.where(label_per_sample == t)[0]) for t in classes])
  weight = 1. / class_sample_count

  print('Stratified sampling weights:')
  print(list(zip(classes, weight)))

  class_idx_per_sample = np.array([class_to_idx[c] for c in label_per_sample])
  samples_weight = np.array([weight[t] for t in class_idx_per_sample])
  sampler = WeightedRandomSampler(list(samples_weight), len(samples_weight) * 2, replacement=True)
  return sampler
