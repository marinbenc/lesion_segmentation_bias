import cv2 as cv
import numpy as np
from medpy.metric.binary import precision as mp_precision
from medpy.metric.binary import recall as mp_recall
from medpy.metric.binary import dc, hd, assd as mp_assd, asd
import matplotlib.pyplot as plt

def thresh(img):
  img[img > 0.5] = 1
  img[img <= 0.5] = 0
  return img

def dsc(y_pred, y_true):
  y_pred = thresh(y_pred)
  y_true = thresh(y_true)

  if not np.any(y_true):
    return 0 if np.any(y_pred) else 1

  score = dc(y_pred, y_true)
  return score

def housdorff_distance(y_pred, y_true):
  y_pred = thresh(y_pred)
  y_true = thresh(y_true)

  if not np.any(y_true):
    return 0 if np.any(y_pred) else 1

  score = hd(y_pred, y_true)
  return score

def assd(y_pred, y_true):
  y_pred = thresh(y_pred)
  y_true = thresh(y_true)

  if not np.any(y_true):
    return 0 if np.any(y_pred) else 1

  assd = np.mean( (asd(y_pred, y_true, None, 1), asd(y_true, y_pred, None, 1)) )
  return assd

def iou(y_pred, y_true):
  y_pred = thresh(y_pred)
  y_true = thresh(y_true)

  intersection = np.logical_and(y_pred, y_true)
  union = np.logical_or(y_pred, y_true)
  if not np.any(union):
    return 0 if np.any(y_pred) else 1
  
  return intersection.sum() / float(union.sum())

def precision(y_pred, y_true):
  #y_pred = _thresh(y_pred).astype(np.int)
  #y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, precision is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.
  
  return mp_precision(y_pred, y_true)

def recall(y_pred, y_true):
  y_pred = thresh(y_pred).astype(np.int32)
  y_true = thresh(y_true).astype(np.int32)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, recall is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.
  
  r = mp_recall(y_pred, y_true)
  return r

def show_torch(imgs, titles=None, show=True, save=False, save_path=None, figsize=(6.4, 4.8), **kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), **kwargs)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if titles is not None:
          axs[0, i].set_title(titles[i])
    if save:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()

def show_images_row(imgs, titles=None, rows=1, figsize=(6.4, 4.8), show=True, **kwargs):
    '''
    Display grid of cv2 images
    :param img: list [cv::mat]
    :param title: titles
    :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    fig = plt.figure(figsize=figsize)
    for n, image in enumerate(imgs):
        ax = fig.add_subplot(rows, int(np.ceil(num_images / float(rows))), n + 1)
        plt.imshow(image, **kwargs)
        plt.axis('off')
    
    if show:
        plt.show()