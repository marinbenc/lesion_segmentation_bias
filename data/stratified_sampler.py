import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch

from data.segmentation_dataset import LesionSegmentationDataset

def StratifiedSampler(dataset: LesionSegmentationDataset):
  """
  Returns a stratified sampler for the dataset. The weights are based on the distribution of the ita_angle attribute in the dataset.
  """
  classes = dataset.all_classes
  labels = dataset.labels_df['label'].to_numpy()
  class_sample_count = np.array([len(np.where(labels == t)[0]) for t in classes])
  weight_for_label = { t: 1. / class_sample_count[i] for i, t in enumerate(classes) }
  samples_weight = [weight_for_label[l] for l in labels]
  sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
  return sampler
