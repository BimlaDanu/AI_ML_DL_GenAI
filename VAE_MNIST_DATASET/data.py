import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=128):
    t=transforms.ToTensor()
    train=datasets.MNIST('./data',train=True,download=True,transform=t)
    test=datasets.MNIST('./data',train=False,download=True,transform=t)
    return (
        DataLoader(train,batch_size=batch_size,shuffle=True),
        DataLoader(test,batch_size=batch_size)
    )

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')