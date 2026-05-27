import timm
import torch.nn as nn
from torchvision import models

def create_model(model_name: str, num_classes: int):
    if model_name in ["vit_tiny_patch16_224", "convnext_tiny"]:
        return timm.create_model(model_name, pretrained = True, num_classes = num_classes)
    else:
        raise ValueError(f"Architecture {model_name} is not supported.")