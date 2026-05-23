import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784,256)
        self.fc_mu = nn.Linear(256,20)
        self.fc_logvar = nn.Linear(256,20)

        self.fc2 = nn.Linear(20,256)
        self.fc3 = nn.Linear(256,784)

    def encode(self,x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        return self.decode(z), mu, logvar


def loss_fn(recon,x,mu,logvar):
    recon_loss = F.binary_cross_entropy(
        recon, x, reduction="sum"
    )
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon_loss + kl