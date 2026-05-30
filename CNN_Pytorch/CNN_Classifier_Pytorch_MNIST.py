import numpy as np
import pandas as pd
import torch
import torchvision


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Device Setting──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Visualisation ─────────────────────────────────────────────────────
def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.grid()
    plt.show()

def visualize_filters(model):
    filters = model.encoder[0].weight.data.clone().cpu()
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    num_filters = min(filters.shape[0], 8)
    fig, axes = plt.subplots(1, num_filters, figsize=(12, 3))
    for i in range(num_filters):
        f = filters[i].permute(1, 2, 0)
        axes[i].imshow(f.squeeze(), cmap='viridis')
        axes[i].axis('off')
    plt.suptitle("First Layer Filters")
    plt.show()

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ── Model ─────────────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 1×28×28 -> 64×7×7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # → 32×14×14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 64×7×7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: 64×7×7 -> 1×28×28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # → 32×14×14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # → 1×28×28
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(device)

# ── Noise ─────────────────────────────────────────────────────────────────────
def add_noise(x, std=0.2):                      # Choose carefully (0.1-1.0)
    return torch.clamp(x + torch.randn_like(x) * std, 0., 1.)

# ── Training Setup ────────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler    = torch.amp.GradScaler('cuda')        # updated deprecated API
epochs    = 50
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# ── Training Loop ─────────────────────────────────────────────────────────────
loss_history = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for img, _ in loader:
        img   = img.to(device, non_blocking=True)
        noisy = add_noise(img)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):        # use updated deprecated API
            output = model(noisy)
            loss   = criterion(output, img)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
images, _ = next(iter(loader))
images       = images[:6].to(device)
noisy_images = add_noise(images)

with torch.no_grad():
    outputs = model(noisy_images)

images       = images.cpu()
noisy_images = noisy_images.cpu()
outputs      = outputs.cpu()

# ── Visualise Results ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
row_labels = ["Original", "Noisy", "Reconstructed"]

for i in range(6):
    axes[0, i].imshow(images[i].squeeze(),       cmap='gray')
    axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
    axes[2, i].imshow(outputs[i].squeeze(),      cmap='gray')
    for row in range(3):
        axes[row, i].axis('off')

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=11)

plt.suptitle("Denoising Autoencoder — MNIST", fontsize=13)
plt.tight_layout()
plt.show()

plot_loss(loss_history)
visualize_filters(model)
