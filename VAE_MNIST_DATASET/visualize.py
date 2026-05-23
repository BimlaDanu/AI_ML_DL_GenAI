import torch
import matplotlib.pyplot as plt

def reconstruction(model,test_loader,device):
    x,_=next(iter(test_loader))
    x=x[:8].view(-1,784).to(device)
    with torch.no_grad():
        recon,_,_=model(x)
    x=x.view(-1,1,28,28).cpu()
    recon=recon.view(-1,1,28,28).cpu()
    fig,ax=plt.subplots(2,8)
    for i in range(8):
        ax[0,i].imshow(x[i][0],cmap='gray'); ax[0,i].axis('off')
        ax[1,i].imshow(recon[i][0],cmap='gray'); ax[1,i].axis('off')
    plt.show()

def generate(model,device):
    with torch.no_grad():
        s=model.decode(torch.randn(16,20).to(device)).view(-1,1,28,28).cpu()
    fig,ax=plt.subplots(4,4)
    for i in range(16):
        ax[i//4,i%4].imshow(s[i][0],cmap='gray')
        ax[i//4,i%4].axis('off')
    plt.show()

def latent(model,test_loader,device):
    mus=[]; labels=[]
    for i,(x,y) in enumerate(test_loader):
        x=x.view(-1,784).to(device)
        mu,_=model.encode(x)
        mus.append(mu[:,:2].cpu())
        labels.append(y)
        if i>10: break
    mus=torch.cat(mus)
    labels=torch.cat(labels)
    plt.scatter(mus[:,0],mus[:,1],c=labels,s=5)
    plt.show()