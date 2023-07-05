from typing import Literal, Optional, Union
import concurrent.futures

from pathlib import Path
import os.path as p
import json

from tqdm import tqdm
import fire
import numpy as np
import pandas as pd
import torch
from skimage import color

from torch.nn.functional import sigmoid

import cv2 as cv

from data.skin_color_dataset import SkinColorDetectionDataset
from models.models import get_detection_model, get_checkpoint
from models.kmeans_skin_color_estimator import find_dominant_color, get_ita_angle, get_fitzpatrick_type

def _process_image_kmeans(file_path):
    img = cv.imread(file_path)
    label_path = file_path.replace('input', 'label').replace('.jpg', '.png')
    if p.exists(label_path):
        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
    else:
        label = None
    dominant_color = find_dominant_color(img, label).squeeze()
    ita_angle = get_ita_angle(dominant_color) - 10.
    fp_type = get_fitzpatrick_type(ita_angle)
    return dominant_color, ita_angle, fp_type

def get_class_from_model_output(output, label_encoding):
    if label_encoding == 'ordinal-2d':
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
    elif label_encoding == 'ordinal-1d':
        output_np = output.squeeze(1).detach().cpu().numpy()
        output_np = np.digitize(output_np, bins=[2/6., 4/6.])
    elif label_encoding == 'class':
        output_np = output.squeeze(1).detach().cpu().numpy()
        output_np = np.argmax(output_np, axis=1)
    else:
        raise NotImplementedError(f'Label encoding {label_encoding} not implemented.')

    return output_np         


def main(
    model_type: Union[Literal['kmeans'], str], 
    dataset_name: Literal['isic', 'dermquest', 'dermis', 'ph2'], 
    colorspace: Literal['rgb', 'lab'] = 'rgb', 
    device: Literal['cpu', 'cuda'] = 'cuda',
    label_encoding: Literal['ordinal-2d', 'ordinal-1d', 'class'] = 'class'):

    dataset = SkinColorDetectionDataset(subset='all', dataset_folder=dataset_name, augment=False, colorspace='rgb', prediction=True, label_encoding=label_encoding)
    predictions_df = dataset.labels_df.copy()
    predictions_df.drop(columns=['file_path'], inplace=True)

    save_path = p.join('data', dataset_name, 'skin_color_prediction.csv')
    if p.exists(save_path):
        print('Loading predictions from:', save_path)
        predictions_df = pd.read_csv(save_path)

    if model_type == 'kmeans':
        colors = np.zeros((len(dataset), 3))
        itas = np.zeros(len(dataset))
        skin_types = np.zeros(len(dataset))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.map(_process_image_kmeans, dataset.labels_df['file_path'])
            for idx, (dominant_color, ita_angle, fp_type) in tqdm(enumerate(results), total=len(dataset)):
                colors[idx] = dominant_color
                itas[idx] = ita_angle
                skin_types[idx] = fp_type

        predictions_df['color_r'] = colors[:, 0]
        predictions_df['color_g'] = colors[:, 1]
        predictions_df['color_b'] = colors[:, 2]
        predictions_df['ita_angle'] = itas
        predictions_df['kmeans_skin_type'] = skin_types
        
        save_path = p.join('data', dataset_name, 'skin_color_prediction.csv')
        predictions_df.to_csv(save_path, index=False)
        print('Saved predictions to:', save_path)
    else:
        data_split = p.join(p.join('runs', model_type, 'subjects.json'))
        with open(data_split, 'r') as f:
            json_dict = json.load(f)
            num_folds = len(json_dict['train_subjects'])

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
        predictions = np.zeros((num_folds, len(dataset)))
        predictions_proba = np.zeros((num_folds, len(dataset), 3))
        for fold in range(num_folds):
            model = get_detection_model(dataset, device)
            model = model.to(device)
            checkpoint = get_checkpoint('skin_detection', model_type, fold, device=device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            
            with torch.no_grad():
                for i, x in tqdm(enumerate(data_loader), total=len(data_loader)):
                    x = x.to(device)
                    model_output = model(x)
                    y_pred = get_class_from_model_output(model_output, label_encoding=label_encoding)
                    predictions[fold, i*16:i*16+len(x)] = y_pred

                    if label_encoding == 'class':
                        y_pred_proba = torch.nn.functional.softmax(model_output, dim=1)
                        y_pred_proba = y_pred_proba.detach().cpu().numpy()
                        predictions_proba[fold, i*16:i*16+len(x)] = y_pred_proba
        
        # soft voting
        # predictions is fold x sample x class
        # predictions_mean = np.mean(predictions, axis=0) # sample x class
        # predictions_class = np.argmax(predictions_mean, axis=1) # sample
        # predictions_class = predictions_class.astype(np.uint8)

        # majority class voting
        predictions_class = np.zeros(len(dataset))
        for sample in range(len(dataset)):
            predictions_class[sample] = np.argmax(np.bincount(predictions[:, sample].astype(np.uint8)))

        predictions_df['nn_skin_type'] = predictions_class
        label_mapping = {0: 12, 1: 34, 2: 56}
        predictions_df['nn_skin_type'] = predictions_df['nn_skin_type'].map(label_mapping)

        for fold in range(num_folds):
            for class_idx in range(3):
                predictions_df[f'f{fold}_c{class_idx}'] = predictions_proba[fold, :, class_idx]
        
        save_path = p.join('data', dataset_name, 'skin_color_prediction.csv')
        predictions_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    fire.Fire(main)