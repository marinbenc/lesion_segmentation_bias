import numpy as np
import argparse
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

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, accuracy_score, classification_report, confusion_matrix

device = 'cuda'

def get_checkpoint(model_type, log_name, fold=0, data_percent=1.):
  checkpoint = p.join('runs', log_name, model_type, f'fold{fold}', f'{model_type}_best_fold={fold}.pth')
  print('Loading checkpoint from:', checkpoint)
  checkpoint = torch.load(checkpoint, map_location=device)
  return checkpoint

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
      print('output_np:', output_np)
           

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
      y_np = [utils._thresh(y) for y in y_np]
      ys += y_np

      xs += [x for x in x_np]

      data = data.to(device)
      output = model.forward(data)

      output_np = output.squeeze(1).detach().cpu().numpy()
      output_np = [utils._thresh(o) for o in output_np]
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

def test(model_type, dataset, log_name, dataset_folder='valid', save_predictions=False, viz=False, label_encoding='ordinal-2d'):
    dataset_args = {
      'subset': dataset_folder,
      'augment': False,
      'colorspace': 'rgb',
      'label_encoding': label_encoding,
       # TODO: Use saved command line arguments / config file saved in train.py
    }
    test_dataset = data.get_dataset_class(dataset)(**dataset_args)

    if model_type == 'skin_detection':
        model = models.get_detection_model(test_dataset, device)
    elif model_type == 'lesion_seg':
        model = models.get_segmentation_model(test_dataset, device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    checkpoint = get_checkpoint(model_type, log_name)
    model.load_state_dict(checkpoint['model'])

    os.makedirs(p.join('predictions', log_name), exist_ok=True)

    if model_type == 'skin_detection':
        xs, ys, ys_pred = get_det_predictions(model, test_dataset, viz=viz)
    elif model_type == 'lesion_seg': 
        xs, ys, ys_pred = get_seg_predictions(model, test_dataset, viz=viz)

        if save_predictions:
            for i in range(len(ys_pred)):
                cv.imwrite(p.join('predictions', log_name, f'{i}.png'), ys_pred[i] * 255)
        
    metrics_seg = {
        'dsc': utils.dsc,
        'hd': utils.housdorff_distance,
        'prec': utils.precision,
        'rec': utils.recall,
    }

    metrics_det = {
        'auc': roc_auc_score,
        'ap': average_precision_score,
        'f1': f1_score,
        'acc': accuracy_score,
    }

    metrics = metrics_seg if model_type == 'lesion_seg' else metrics_det
    #df = calculate_metrics(ys, ys_pred, metrics, subjects=test_dataset.subject_id_for_idx)

    #df.to_csv(p.join('predictions', log_name, 'metrics.csv'))
    #if test_dataset.subject_id_for_idx is not None:
    #  df = df.groupby('subject').mean()

    #print(df.describe())

    cm = confusion_matrix(ys, ys_pred)
    print(cm)

    print('accuracy:', cm.diagonal() / cm.sum(axis=1))

    print(classification_report(ys, ys_pred, target_names=['12', '34', '56']))
    #return df

if __name__ == '__main__':
    fire.Fire(test)