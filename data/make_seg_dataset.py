import sys
import os
import numpy as np
from shutil import copyfile
from pathlib import Path
from glob import glob

from torch.utils.data import random_split
import cv2 as cv

sys.path.append('../../')

import PIL.Image

def get_files(folder):
  files = os.listdir(folder)
  files.sort()
  files = [f for f in files if not '.txt' in f]
  files = [os.path.join(folder, f) for f in files]
  return files

def save_files(files, folder):
  os.makedirs(os.path.join(folder, 'input'), exist_ok=True)
  os.makedirs(os.path.join(folder, 'label'), exist_ok=True)

  for input_file, gt_file in files:
    file_name = input_file.split('/')[-1]
    input_destination = os.path.join(folder, 'input', file_name)
    gt_destionation = os.path.join(folder, 'label', file_name.replace('.jpg', '.png'))

    input_img = cv.imread(input_file)
    gt_img = cv.imread(gt_file, cv.IMREAD_GRAYSCALE)
    gt_img = np.array(PIL.Image.fromarray(gt_img).resize((256, 256), PIL.Image.NEAREST)) # OpenCV INTER_NEAREST has a bug
    input_img = cv.resize(input_img, (256, 256), interpolation=cv.INTER_LINEAR)
    
    cv.imwrite(input_destination, input_img)
    cv.imwrite(gt_destionation, gt_img)

wd = Path(os.path.dirname(os.path.realpath(__file__)))

dataset_groups = ['isic', 'dermis', 'dermquest']
for group in dataset_groups:
  os.makedirs(wd/group, exist_ok=True)

  if group == 'isic':
    VALID_INPUT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1-2_Validation_Input'
    TRAIN_INPUT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1-2_Training_Input'
    VALID_GT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1_Validation_GroundTruth'
    TRAIN_GT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1_Training_GroundTruth'

    valid_input = get_files(wd/VALID_INPUT_FOLDER)
    train_input = get_files(wd/TRAIN_INPUT_FOLDER)

    valid_gt = get_files(wd/VALID_GT_FOLDER)
    train_gt = get_files(wd/TRAIN_GT_FOLDER)

    inputs = valid_input + train_input
    gts = valid_gt + train_gt
  
  elif group in ['dermis', 'dermquest']:
    subset = group.split('_')[-1]
    subset_folder = subset
    if subset_folder == 'dermis':
      subset_folder = 'dermIS'
    
    gts = glob(f'{wd}/downloaded_data/waterloo/**/*_contour.png', recursive=True)
    gts = [gt for gt in gts if subset_folder in gt or subset in gt]

    inputs = [f.replace('_contour.png', '_orig.jpg') for f in gts]
  
  # split same as in Double U-Net paper: https://arxiv.org/pdf/2006.04868v2.pdf
  train_valid_test_split = (0.8, 0.1, 0.1)

  test_count = int(train_valid_test_split[2] * len(inputs))
  valid_count = test_count
  train_count = len(inputs) - test_count * 2

  print(group, train_count, valid_count, test_count)
  assert(test_count + valid_count + train_count == len(inputs))

  all_files = np.array(list(zip(inputs, gts)))
  np.random.seed(2022)
  np.random.shuffle(all_files)

  train_files = all_files[:train_count]
  valid_files = all_files[train_count : train_count + valid_count]
  test_files = all_files[-test_count:]

  save_files(train_files, wd/group/'train')
  save_files(valid_files, wd/group/'valid')
  save_files(test_files, wd/group/'test')