from torchvision import transforms

def build_transforms(preprocessing_config: dict):

    size = preprocessing_config['image_size']
    mean = preprocessing_config['mean']
    std = preprocessing_config['std']
    
    # in case a single value was provided in config, 
    # it is expanded into three (as we work with RGB images)
    if len(mean) == 1:
        mean = mean * 3
        std = std * 3

    # defines sequential preprocessing pipeline
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])