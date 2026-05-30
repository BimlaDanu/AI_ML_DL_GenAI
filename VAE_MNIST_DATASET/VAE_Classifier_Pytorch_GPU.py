!nvidia-smi
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm 

# ------- VARIATIONAL AUTOENCODER (VAE) — MNIST ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Device Setting──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# ── Data Loading──────────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# ── VAE Model Construction─────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),             # stabilises training
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            #nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * std  (eps ~ N(0,I))  — differentiable via mu/std"""
        if self.training:                           # only add noise during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu                                   # use mean directly at eval time

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model     = VAE(input_dim=784, hidden_dim=512, latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
scaler    = torch.amp.GradScaler('cuda')            # mixed precision

# ── Loss Function ─────────────────────────────────────────────────────────────
def loss_fn(recon, x, mu, logvar, beta=1.0):
    """
    ELBO loss = Reconstruction loss + β * KL divergence
    BCE  : measures pixel-wise reconstruction quality
    KL   : pushes posterior q(z|x) toward prior N(0,I)
    beta : β > 1 encourages more disentangled latent space (β-VAE)
    """
    #bce = F.binary_cross_entropy(recon, x, reduction='sum')
    #kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return (bce + beta * kl) / x.size(0)           # normalise by batch size
    bce = F.binary_cross_entropy_with_logits(recon, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + beta * kl) / x.size(0)

# ── Training Loop ─────────────────────────────────────────────────────────────
epochs           = 50
beta             = 1.0
train_loss_hist  = []
test_loss_hist   = []

for epoch in range(epochs):

    # — Train —
    model.train()
    total_train = 0.0

    for x, _ in train_loader:
        x = x.view(-1, 784).to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar, beta)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train += loss.item()

    avg_train = total_train / len(train_loader)
    train_loss_hist.append(avg_train)

    # — Test —
    model.eval()
    total_test = 0.0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(-1, 784).to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(x)
                loss = loss_fn(recon, x, mu, logvar, beta)
            total_test += loss.item()

    avg_test = total_test / len(test_loader)
    test_loss_hist.append(avg_test)
    scheduler.step(avg_test)

    print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")

# ──Reconstruction ────────────────────────────────────────────────────
model.eval()
x, _ = next(iter(test_loader))
x_sample = x[:8].view(-1, 784).to(device)

with torch.no_grad():
    recon, _, _ = model(x_sample)

orig  = x_sample.view(-1, 1, 28, 28).cpu()
recon = recon.view(-1, 1, 28, 28).cpu()

fig, ax = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    ax[0, i].imshow(orig[i][0],  cmap="gray")
    ax[1, i].imshow(recon[i][0], cmap="gray")
    for row in range(2):
        ax[row, i].axis("off")
ax[0, 0].set_ylabel("Original",     fontsize=9)
ax[1, 0].set_ylabel("Reconstructed",fontsize=9)
plt.suptitle("Test 1 — Reconstruction", fontsize=12)
plt.tight_layout()
plt.show()

# ──Random Generation from Prior ─────────────────────────────────────
with torch.no_grad():
    z       = torch.randn(16, 20).to(device)
    #samples = model.decode(z).view(-1, 1, 28, 28).cpu()
    samples = torch.sigmoid(model.decode(z)).view(-1, 1, 28, 28).cpu()

fig, ax = plt.subplots(4, 4, figsize=(5, 5))
for i in range(16):
    ax[i // 4, i % 4].imshow(samples[i][0], cmap="gray")
    ax[i // 4, i % 4].axis("off")
plt.suptitle("Test 2 — Generated Samples (z ~ N(0,I))", fontsize=11)
plt.tight_layout()
plt.show()

# ──Latent Space 2D Visualisation ────────────────────────────────────
def plot_latent_space(model, loader, num_batches=40):
    model.eval()
    mus_list    = []
    labels_list = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x  = x.view(-1, 784).to(device)
            mu, _ = model.encode(x)
            mus_list.append(mu[:, :2].cpu())        # FIX: .cpu() before .numpy()
            labels_list.append(y)

    mus    = torch.cat(mus_list).numpy()
    labels = torch.cat(labels_list).numpy()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=3, alpha=0.7)
    plt.colorbar(sc, ticks=range(10), label="Digit")
    plt.title("Test 3 — Latent Space 2D Projection (first 2 dims of μ)")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.tight_layout()
    plt.show()

plot_latent_space(model, test_loader)

# ──Latent Space Interpolation ───────────────────────────────────────
def plot_interpolation(model, loader, steps=10):
    """Linearly interpolate between the latent codes of two test images."""
    model.eval()
    x, y = next(iter(loader))

    # pick two different digits
    idx_a = (y == 3).nonzero(as_tuple=True)[0][0]
    idx_b = (y == 7).nonzero(as_tuple=True)[0][0]

    xa = x[idx_a].view(1, 784).to(device)
    xb = x[idx_b].view(1, 784).to(device)

    with torch.no_grad():
        mu_a, _ = model.encode(xa)
        mu_b, _ = model.encode(xb)

        alphas = torch.linspace(0, 1, steps).to(device)
        imgs   = []
        for alpha in alphas:
            z     = (1 - alpha) * mu_a + alpha * mu_b   # linear interpolation
            img   = model.decode(z).view(1, 28, 28).cpu()
            imgs.append(img)

    fig, ax = plt.subplots(1, steps, figsize=(steps * 1.2, 2))
    for i, img in enumerate(imgs):
        ax[i].imshow(img[0], cmap="gray")
        ax[i].axis("off")
    ax[0].set_title("'3'",  fontsize=9)
    ax[-1].set_title("'7'", fontsize=9)
    plt.suptitle("Test 4 — Latent Interpolation: 3 -> 7", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_interpolation(model, test_loader)

# ──Per-Digit Reconstruction─────────────────────────────────
def per_digit_loss(model, loader):
    model.eval()
    digit_loss  = torch.zeros(10)
    digit_count = torch.zeros(10)

    with torch.no_grad():
        for x, y in loader:
            x = x.view(-1, 784).to(device)
            recon, mu, logvar = model(x)
            recon_prob = torch.sigmoid(recon)          # FIX: convert logits → probs first

            for d in range(10):
                mask = (y == d)
                if mask.sum() == 0:
                    continue
                x_d     = x[mask]
                recon_d = recon_prob[mask]
                bce     = F.binary_cross_entropy(recon_d, x_d, reduction='sum')  # safe now
                digit_loss[d]  += bce.cpu()
                digit_count[d] += mask.sum()

    print("\nPer-Digit Reconstruction Loss (BCE / pixel):")
    for d in range(10):
        avg = digit_loss[d] / (digit_count[d] * 784)
        print(f"  Digit {d}: {avg:.4f}")

    plt.figure(figsize=(7, 4))
    avg_losses = [(digit_loss[d] / (digit_count[d] * 784)).item() for d in range(10)]
    plt.bar(range(10), avg_losses, color="steelblue")
    plt.xlabel("Digit")
    plt.ylabel("Avg BCE per pixel")
    plt.title("Test 5 — Per-Digit Reconstruction Loss")
    plt.xticks(range(10))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()