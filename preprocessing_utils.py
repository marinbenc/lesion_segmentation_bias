import warnings
from typing import Optional, Tuple, Literal

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

# color.lab2rgb() warns when values are clipped, but that is not a problem here
warnings.filterwarnings('ignore', message='.*values that have been clipped.*', append=True)

def equalize_hist(img, white_point_color):
    #white_point_color_norm = white_point_color / np.linalg.norm(white_point_color)
    #white_point_color_norm *= 255
    white_point_color = cv.cvtColor(np.uint8([[white_point_color]]), cv.COLOR_RGB2LAB)[0][0]
    white_l = white_point_color[0]
    img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    img_lab_l = img_lab[:,:,0]
    hist, bins = np.histogram(img_lab_l.flatten(), 256, (0, 256))
    # threshold
    hist[white_l:] = 0
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img_lab_l = cdf[img_lab_l]
    img_lab[:,:,0] = img_lab_l
    img = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    return img

def shade_of_gray_cc(img: np.ndarray, power: float = 6, gamma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Color constancy algorithm based on Shades of Gray method.
    based on: https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy
    C. Barata, M. E. Celebi and J. S. Marques, 
    "Improving Dermoscopy Image Classification Using Color Constancy," 
    in IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 3, pp. 1146-1152, May 2015, 
    doi: 10.1109/JBHI.2014.2336473.

    Args:
        img (numpy array): the original image with format of (h, w, c)
        power (int): the degree of norm, 6 is used in reference paper
        gamma (float): the value of gamma correction, 2.2 is used in reference paper
    Returns:
        img: img after color constancy
        rgb_vec: illuminant applied to img
        color: the estimated skin color in rgb
    """
    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    # gamma correct rgb_vec
    rgb_vec = np.power(rgb_vec / 255., 1/1.3) * 255.
    color = rgb_vec.copy()
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    
    return img, rgb_vec, color

def illuminant_from_color(color_rgb):
    rgb_vec = color_rgb.copy()
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/(rgb_norm + 1e-6)
    rgb_vec = 1/(rgb_vec*np.sqrt(3)+1e-6)
    return rgb_vec

def _float_inputs(lab1, lab2, allow_float32=True):
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    lab1 = lab1.astype(np.float32, copy=False)
    lab2 = lab2.astype(np.float32, copy=False)
    return lab1, lab2

def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """
    Based on: https://github.com/scikit-image/scikit-image/blob/main/skimage/color/delta_e.py
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    def distance(x, y):
        return np.sqrt((x - y) ** 2)
    
    dist = np.array([distance(L1, L2), distance(a1, a2), distance(b1, b2)])
    return dist.transpose(1, 2, 0)

def dist_img(img, skin_color):
    img_hsv = color.rgb2hsv(img)
    black_regions = np.logical_and(img_hsv[:, :, 1] < 0.1, img_hsv[:, :, 2] < 0.1)
    img_ = img.copy()
    img_[black_regions] = skin_color
    img_lab = color.rgb2lab(img_)
    skin_color_lab = color.rgb2lab(skin_color)
    dist = deltaE_cie76(img_lab, np.ones_like(img_lab) * skin_color_lab) # TODO: Use per-channel distance
    radius = img.shape[0] // 2
    center = (img.shape[0] // 2, img.shape[1] // 2)
    circle_mask = cv.circle(np.zeros_like(img), center, radius, (1, 1, 1), -1)
    dist_masked = dist * circle_mask
    # normalize each channel separately
    for i in range(3):
        max_dist = np.max(dist_masked[:, :, i])
        dist_c = dist[:, :, i]
        dist_c[dist_c > max_dist] = max_dist
        dist[:, :, i] = dist_c / max_dist
        dist[:, :, i] = equalize_adapthist(dist[:, :, i])
    return dist
