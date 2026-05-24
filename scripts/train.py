from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import argparse
import torch
import yaml
import os

from src.data_transform import build_transforms
from src.data_loader import get_dataloaders
from src.models.factory import create_model
from src.trainer import train_model


def main():

    # parsing arguments from yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # if drive is not mounted, we store data locally
    if os.path.exists("/content/drive/MyDrive"):
        config['log_dir'] = "/content/drive/MyDrive/colorectal_cancer_project/logs"
    else:
        config['log_dir'] = "logs"
    
    # setting up device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Initializing {config['model_name']} on {device}...")

    # tranforming data and creating the model using factory.py
    transform_pipeline = build_transforms(config['preprocessing'])
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        transform=transform_pipeline)
    model = create_model(config['model_name'], config['num_classes']).to(device)
    
    # setting up parameters and starting training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max = config['epochs'], eta_min = 1e-6)
    train_model(model, train_loader, test_loader, device, criterion, optimizer, scheduler, config)

if __name__ == "__main__":
    main()