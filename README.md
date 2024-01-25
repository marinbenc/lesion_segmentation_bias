## Understanding skin color bias in deep learning-based skin lesion segmentation

This is the code repository for the paper

Marin Benčević, Marija Habijan, Irena Galić, Danilo Babin, Aleksandra Pižurica,
Understanding skin color bias in deep learning-based skin lesion segmentation,
Computer Methods and Programs in Biomedicine, 2024, 108044, ISSN 0169-2607, https://doi.org/10.1016/j.cmpb.2024.108044.

https://www.sciencedirect.com/science/article/pii/S0169260724000403

```
@article{Bencevic2024Understanding,
title = {Understanding skin color bias in deep learning-based skin lesion segmentation},
journal = {Computer Methods and Programs in Biomedicine},
pages = {108044},
year = {2024},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2024.108044},
url = {https://www.sciencedirect.com/science/article/pii/S0169260724000403},
author = {Marin Benčević and Marija Habijan and Irena Galić and Danilo Babin and Aleksandra Pižurica},
}
```

## Data Preparatation

The data links are below. Each dataset can be downloaded and extracted into `data/downloaded_data`, an example of the folder structure:

```
data/downloaded_data
├── dermofit
│   ├── ak
│   ├── bcc
│   ├── df
│   ├── iec
│   ├── lesionlist.txt
│   ├── mel
│   ├── ml
│   ├── pyo
│   ├── scc
│   ├── sk
│   └── vasc
├── diverse
│   └── ddidiversedermatologyimages
├── fp17k
│   ├── fitzpatrick17k.csv
│   └── images
├── isic
│   ├── ISIC2018_Task1-2_Test_Input
│   ├── ISIC2018_Task1-2_Training_Input
│   ├── ISIC2018_Task1-2_Validation_Input
│   ├── ISIC2018_Task1_Test_GroundTruth
│   ├── ISIC2018_Task1_Training_GroundTruth
│   └── ISIC2018_Task1_Validation_GroundTruth
├── pad_ufes_20
│   └── zr7vgbcyr2-1
├── ph2
│   └── PH2Dataset
└── waterloo
    ├── Skin Image Data Set-1
    └── Skin Image Data Set-2
```

To split and prepare the segmentation images, run:

```
cd data
python make_seg_dataset.py
```

To split and prepare the skin color prediction images, run:

```
cd data
python make_skin_color_dataset.py
```

### Segmentation Datasets

**ISIC 2018**

https://challenge.isic-archive.com/data/#2018

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

**Waterloo**

https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection

**Dermofit**

https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library

Rees, Aldridge, Fisher, Ballerini (2013), A Color and Texture Based Hierarchical K-NN Approach to the Classification of Non-melanoma Skin Lesions, Color Medical Image Analysis, Lecture Notes in Computational Vision and Biomechanics 6 (M. E. Celebi, G. Schaefer (eds.))

**PH2**

https://www.fc.up.pt/addi/ph2%20database.html

Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira. PH² - A dermoscopic image database for research and benchmarking, 35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

### Skin Color Estimation Datasets

**Diverse**

https://ddi-dataset.github.io/

Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set. Roxana Daneshjou, Kailas Vodrahalli, Weixin Liang, Roberto A Novoa, Melissa Jenkins, Veronica Rotemberg, Justin Ko, Susan M Swetter, Elizabeth E Bailey, Olivier Gevaert, Pritam Mukherjee, Michelle Phung, Kiana Yekrang, Bradley Fong, Rachna Sahasrabudhe, James Zou, Albert Chiou. Science Advances (2022) 

**Fitzpatrick 17k**

https://github.com/mattgroh/fitzpatrick17k

**PAD UFES**

https://www.sciencedirect.com/science/article/pii/S235234092031115X

## Model Training

To train a model, run `python train.py -h` to see the available options.

## Model Testing

Run `python test.py` to see options. Test results will be saved in `predictions/LOG_NAME/metrics_TEST_DATASET_NAME.csv`.

## Predicting Skin Colors

To run the skin color prediction, first train a skin color prediction model using the above command, then run

```
python predict_fp_skin_type.py MODEL_TYPE DATASET_NAME <flags> (see -h for options)
```

If `MODEL_TYPE` is `kmeans`, then the skin prediction will be run using the image processing-based method. Otherwise, `MODEL_TYPE` should be the log name used to train the skin prediction model. The results will be saved in `data/DATASET_NAME/skin_color_prediction.csv`.

## Pretrained Models

All models used in the analysis are available in the Releases section. The folders should be places in `runs/`. Each model includes the 5-fold subject split and the command that was used to obtain the model, the trained weights as well as a Tensorboard log.