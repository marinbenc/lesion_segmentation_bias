import torch
import segmentation_models_pytorch as smp

def get_segmentation_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    unet = smp.Unet('resnet18', in_channels=3, classes=1, 
                    activation='sigmoid', decoder_use_batchnorm=True)
    unet = unet.to(device)
    if checkpoint is not None:
      saved_unet = torch.load(checkpoint)
      unet.load_state_dict(saved_unet['model'])
    return unet

def get_detection_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    # TODO: Resnet18 detection model
    raise NotImplementedError
