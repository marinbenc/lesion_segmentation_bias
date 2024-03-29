import numpy as np
import json
from glob import glob
import torch
from torch.nn.functional import softmax, sigmoid
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import os.path as p

from torch.utils.data import DataLoader

import pandas as pd
import fire

import utils
import data.datasets as data
import models.models as models

from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, 
                            f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score)

device = 'cuda'

def get_det_predictions(model, dataset, viz=True):
  xs = []
  ys = []
  ys_pred = []

  loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

  model.eval()
  with torch.no_grad():
    for (data, target) in tqdm(loader):
      x_np = data.squeeze(1).detach().cpu().numpy()
      xs += [x for x in x_np]

      data = data.to(device)
      output = model(data)

      if dataset.label_encoding == 'ordinal-2d':
        y_np = target.detach().cpu().numpy().astype(np.int8)
        y_np = y_np.sum(axis=1) - 1
        y_np = [y for y in y_np]
        ys += [y for y in y_np]

        output_ = sigmoid(output)
        output = output_ > 0.5

        output_np = np.zeros(output.shape[0])
        # (index of first 0) - 1, e.g. [1, 1, 0] => 1 (class 1)
        for batch in range(output.shape[0]):
          output_batch = output[batch].detach().cpu().numpy()
          if output_batch[2] == True:
              output_np[batch] = 2
          elif output_batch[1] == True:
              output_np[batch] = 1
          elif output_batch[0] == True:
              output_np[batch] = 0
          else:
              output_np[batch] = -1
      elif dataset.label_encoding == 'ordinal-1d':
        y_np = target.squeeze(1).detach().cpu().numpy()
        y_np = np.digitize(y_np, bins=[2/6., 4/6.])
        ys += [y for y in y_np]
        output_np = output.squeeze(1).detach().cpu().numpy()
        output_np = np.digitize(output_np, bins=[2/6., 4/6.])
      elif dataset.label_encoding == 'class':
        y_np = target.detach().cpu().numpy()
        ys += [y for y in y_np]
        output_np = output.squeeze(1).detach().cpu().numpy()
        output_np = np.argmax(output_np, axis=1)           

      ys_pred += [o for o in output_np]

  # print ys_pred counts
  print('ys_pred counts:', np.unique(ys_pred, return_counts=True))
  return xs, ys, ys_pred


def get_seg_predictions(model, dataset, viz=True):
  xs = []
  ys = []
  ys_pred = []

  loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
  model.eval()
  with torch.no_grad():
    for (data, target) in tqdm(loader):
      x_np = data.squeeze(1).detach().cpu().numpy()
      y_np = target['seg'].squeeze(1).detach().cpu().numpy()
      y_np = [utils.thresh(y) for y in y_np]
      ys += y_np

      xs += [x for x in x_np]

      data = data.to(device)
      output = model.forward(data)

      output_np = output.squeeze(1).detach().cpu().numpy()
      output_np = [utils.thresh(o) for o in output_np]
      ys_pred += [o for o in output_np]

      if viz and y_np[0].sum() > 5:
        utils.show_torch(imgs=[target['seg'][0].squeeze(), output[0].squeeze()], titles=['target', 'output'])

  return xs, ys, ys_pred

def calculate_metrics(ys_pred, ys, metrics, subjects=None):
  '''
  Parameters:
    ys_pred: model-predicted segmentation masks
    ys: the GT segmentation masks
    metrics: a dictionary of type `{metric_name: metric_fn}` 
    where `metric_fn` is a function that takes `(y_pred, y)` and returns a float.
    subjects: a list of subject IDs, one for each element in ys_pred. If provided, the
    returned DataFrame will have a column with the subject IDs.

  Returns:
    A DataFrame with one column per metric and one row per image.
  '''
  metric_names, metric_fns = list(metrics.keys()), metrics.values()
  columns = metric_names + ['subject']
  df = pd.DataFrame(columns=columns)

  if subjects is None:
    subjects = ['none'] * len(ys_pred)

  df['subject'] = subjects
  df['subject'] = df['subject'].astype('category')
  df.set_index(keys='subject', inplace=True)
  for (metric_name, fn) in metrics.items():
    df[metric_name] = [fn(y_pred, y) for (y_pred, y) in zip(ys_pred, ys)]
  
  return df

def test(model_type, dataset, log_name, dataset_folder=None, save_predictions=False, viz=False, label_encoding='ordinal-2d', colorspace='rgb'):
    ys, ys_pred, subject_ids = [], [], []

    datasets = []
    if dataset_folder is None:
      data_split = p.join(p.join('runs', log_name, 'subjects.json'))
      with open(data_split, 'r') as f:
        json_dict = json.load(f)
        splits = zip(json_dict['train_subjects'], json_dict['valid_subjects'])

      for fold, split in enumerate(splits):
        valid_subjects = split[1]
        dataset_args = {
          'subset': 'all',
          'augment': False,
          'colorspace': colorspace,
          # TODO: Use saved command line arguments / config file saved in train.py
        }

        if model_type == 'skin_detection':
          dataset_args['label_encoding'] = label_encoding

        test_dataset = data.get_dataset_class(dataset)(**dataset_args, subjects=valid_subjects)
        datasets.append(test_dataset)
    else:
      dataset_args = {
        'subset': dataset_folder,
        'augment': False,
        'colorspace': colorspace,
      }

      test_dataset = data.get_dataset_class(dataset)(**dataset_args)
      datasets.append(test_dataset)

    for fold, test_dataset in enumerate(datasets):
      if model_type == 'skin_detection':
          model = models.get_detection_model(test_dataset, device)
      elif model_type == 'lesion_seg':
          model = models.get_segmentation_model(test_dataset, device)
      else:
          raise ValueError(f'Unknown model type: {model_type}')

      checkpoint = models.get_checkpoint(model_type, log_name, fold)
      model.load_state_dict(checkpoint['model'])

      os.makedirs(p.join('predictions', log_name), exist_ok=True)

      if model_type == 'skin_detection':
          xs, ys_fold, ys_pred_fold = get_det_predictions(model, test_dataset, viz=viz)
          subject_ids_fold = []
      elif model_type == 'lesion_seg': 
          xs, ys_fold, ys_pred_fold = get_seg_predictions(model, test_dataset, viz=viz)
          subject_ids_fold = test_dataset.subject_id_for_idx

          if save_predictions:
              for i in range(len(ys_pred_fold)):
                  file_name = test_dataset.file_names[i].split('/')[-1].replace('.jpg', '.png')
                  cv.imwrite(p.join('predictions', log_name, file_name), ys_pred_fold[i] * 255)
      else:
          raise ValueError(f'Unknown model type: {model_type}')

      ys += list(ys_fold)
      ys_pred += list(ys_pred_fold)
      subject_ids += list(subject_ids_fold)
        
    metrics = {
        'dsc': utils.dsc,
        'hd': utils.housdorff_distance,
        'prec': utils.precision,
        'rec': utils.recall,
    }

    if model_type == 'lesion_seg':
      metrics = {
          'dsc': utils.dsc,
          'hd': utils.housdorff_distance,
          'assd': utils.assd,
          'prec': utils.precision,
          'rec': utils.recall,
      }

      df = calculate_metrics(ys, ys_pred, metrics, subjects=subject_ids)

      df.to_csv(p.join(f'predictions', log_name, f'metrics_{dataset}.csv'))
      print(df.describe())
    else:
      cm = confusion_matrix(ys, ys_pred)
      print(cm)

      # balanced accuracy
      print('balanced accuracy:', balanced_accuracy_score(ys, ys_pred))

      print(classification_report(ys, ys_pred, target_names=['12', '34', '56']))

if __name__ == '__main__':
    fire.Fire(test)