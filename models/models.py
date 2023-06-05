import torch
import segmentation_models_pytorch as smp

from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

def get_segmentation_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    unet = smp.Unet('resnet18', in_channels=3, classes=1, 
                    activation='sigmoid', decoder_use_batchnorm=True)
    unet = unet.to(device)
    if checkpoint is not None:
      saved_unet = torch.load(checkpoint)
      unet.load_state_dict(saved_unet['model'])
    return unet

def get_detection_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    num_classes = len(dataset.all_classes)
    model.fc = torch.nn.Linear(num_features, num_classes)
    model = model.to(device)
    if checkpoint is not None:
        saved_model = torch.load(checkpoint)
        model.load_state_dict(saved_model['model'])
    return model
