import timm
import torch.nn as nn
from torchvision import models

def create_model(model_name: str, num_classes: int):
    if "vit" in model_name:
        return timm.create_model(model_name, pretrained = True, num_classes = num_classes)

    ## MORE MODELS HERE AS ELIF

    else:
        raise ValueError(f"Architecture {model_name} is not supported.")