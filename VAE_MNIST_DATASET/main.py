import torch.optim as optim
from data import get_loaders,get_device
from model import VAE
from train import train
from visualize import reconstruction,generate,latent

def main():
    device=get_device()
    print('device=',device)
    train_loader,test_loader=get_loaders()
    model=VAE().to(device)
    opt=optim.Adam(model.parameters(),lr=1e-3)
    train(model,train_loader,opt,device,epochs=12)
    reconstruction(model,test_loader,device)
    generate(model,device)
    latent(model,test_loader,device)

if __name__=='__main__':
    main()