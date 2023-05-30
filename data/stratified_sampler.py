import numpy as np
from torch.utils.data import WeightedRandomSampler

from data.segmentation_dataset import LesionSegmentationDataset

def StratifiedSampler(dataset: LesionSegmentationDataset):
  """
  Returns a stratified sampler for the dataset. The weights are based on the distribution of the ita_angle attribute in the dataset.
  """
  ita_angle = dataset.ita_angles.copy()

  bins = 6
  hist = np.histogram(ita_angle, bins=bins, density=True)
  weights = hist[0] / np.sum(hist[0])
  weights = 1 / weights
  weights = weights / np.sum(weights)

  print('Stratified sampling:')
  print('Weights:', weights)

  bin_per_image = np.digitize(ita_angle, hist[1], right=True)
  sample_weights = np.zeros(len(dataset))
  for i in range(len(dataset)):
    sample_weights[i] = weights[bin_per_image[i] - 1]
  
  return WeightedRandomSampler(list(sample_weights), len(sample_weights) * 2, replacement=True)
