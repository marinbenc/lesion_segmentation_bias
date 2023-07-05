import torch
import segmentation_models_pytorch as smp
import os.path as p

from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, vgg16_bn, VGG16_BN_Weights, vgg16, VGG16_Weights

def get_checkpoint(model_type, log_name, fold=0, data_percent=1., device='cuda'):
  checkpoint = p.join('runs', log_name, model_type, f'fold{fold}', f'{model_type}_best_fold={fold}.pth')
  print('Loading checkpoint from:', checkpoint)
  checkpoint = torch.load(checkpoint, map_location=device)
  return checkpoint

def get_segmentation_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    unet = smp.Unet('resnet18', in_channels=3, classes=1, 
                    activation='sigmoid', decoder_use_batchnorm=True)
    unet = unet.to(device)
    if checkpoint is not None:
      saved_unet = torch.load(checkpoint)
      unet.load_state_dict(saved_unet['model'])
    return unet

def get_detection_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # num_classes = len(dataset.all_classes)
    # if dataset.label_encoding == 'class' or dataset.label_encoding == 'ordinal-2d':
    #     model.fc = torch.nn.Linear(num_features, num_classes)
    # elif dataset.label_encoding == 'ordinal-1d':
    #     model.fc = torch.nn.Linear(num_features, 1)
    # else:
    #     raise ValueError(f'Unknown label encoding: {dataset.label_encoding}')
    # model = model.to(device)

    model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    num_classes = len(dataset.all_classes)
    if dataset.label_encoding == 'class' or dataset.label_encoding == 'ordinal-2d':
        model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    elif dataset.label_encoding == 'ordinal-1d':
        model.classifier[6] = torch.nn.Linear(num_features, 1)
    model = model.to(device)
    if checkpoint is not None:
        saved_model = torch.load(checkpoint)
        model.load_state_dict(saved_model['model'])
    return model
