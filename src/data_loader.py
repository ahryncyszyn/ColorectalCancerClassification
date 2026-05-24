from torch.utils.data import DataLoader
from medmnist import PathMNIST

def get_dataloaders(batch_size: int, transform):
    
    # maps dataset indices to the hard drive and registers the transform pipeline
    train_dataset = PathMNIST(split = 'train', transform = transform, download = True)
    # creates an iterator that fetches images, applies transforms, and groups
    # them into [batch_size] tensors during the training loop
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    test_dataset = PathMNIST(split = 'test', transform = transform, download = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader