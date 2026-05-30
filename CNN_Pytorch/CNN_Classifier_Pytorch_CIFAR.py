import numpy as np
import pandas as pd
import torch
import torchvision

# ===== CIFAR-10 CNN CLASSIFICATION WITH NORMALIZATION =====

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Device Setting──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Visualisation Function ─────────────────────────────────────────────────────
def plot_loss(train_history, test_history):
    plt.figure()
    plt.plot(train_history, label="Train Loss")
    plt.plot(test_history,  label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_filters(model):
    filters = model.conv[0].weight.data.clone().cpu()
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    num_filters = min(filters.shape[0], 8)
    fig, axes = plt.subplots(1, num_filters, figsize=(12, 3))
    for i in range(num_filters):
        f = filters[i].permute(1, 2, 0)   # CHW → HWC
        axes[i].imshow(f)
        axes[i].axis('off')
    plt.suptitle("First Layer Filters")
    plt.show()

def unnormalize(img):
    return torch.clamp(img * 0.5 + 0.5, 0., 1.)   # reverse [-1,1] → [0,1]

# ── Data ──────────────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),              # augmentation
    transforms.RandomCrop(32, padding=4),           # augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform_train)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ── Model ─────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),                     #  change for speeds convergence
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # -> 32×16×16

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # -> 64×8×8

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # -> 128×4×4
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                        # this helps in reducing overfitting
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN().to(device)                            # Move model to GPU

# ── Training Setup ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularisation
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
scaler    = torch.amp.GradScaler('cuda')            # mixed precision

# ── Training Loop ─────────────────────────────────────────────────────────────
epochs = 100
train_loss_history = []
test_loss_history  = []

for epoch in range(epochs):

    # — Train —
    model.train()
    running_loss = 0.0

    for img, label in train_loader:
        img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

        optimizer.zero_grad()                       # Moved before forward pass

        with torch.amp.autocast('cuda'):            # Updated deprecated API
            output = model(img)
            loss   = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # — Validation —
    model.eval()
    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                output = model(img)
                loss   = criterion(output, label)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total   += label.size(0)
            correct += (predicted == label).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_loss_history.append(avg_test_loss)
    scheduler.step(avg_test_loss)

    print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} "
          f"| Test Loss: {avg_test_loss:.4f} | Acc: {100*correct/total:.2f}%")

# ── Final Evaluation + Per-Class Accuracy ────────────────────────────────────
model.eval()
class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        for c in range(10):
            mask = (label == c)
            class_correct[c] += (predicted[mask] == label[mask]).sum().item()
            class_total[c]   += mask.sum().item()

print("\nPer-Class Accuracy:")
for c in range(10):
    print(f"  {classes[c]:12s}: {100 * class_correct[c] / class_total[c]:.1f}%")
overall = 100 * sum(class_correct) / sum(class_total)
print(f"\nOverall Test Accuracy: {overall:.2f}%")

# ── Visualise Predictions ─────────────────────────────────────────────────────
images, labels = next(iter(test_loader))
images, labels = images[:6].to(device), labels[:6].to(device)

with torch.no_grad():
    _, preds = torch.max(model(images), 1)

images = images.cpu()
preds  = preds.cpu()
labels = labels.cpu()

fig, axes = plt.subplots(1, 6, figsize=(12, 3))
for i in range(6):
    img = unnormalize(images[i]).permute(1, 2, 0)
    color = 'green' if preds[i] == labels[i] else 'red'
    axes[i].imshow(img)
    axes[i].set_title(f"P: {classes[preds[i]]}\nT: {classes[labels[i]]}", color=color)
    axes[i].axis('off')

plt.suptitle("Green = Correct  |  Red = Wrong", fontsize=11)
plt.tight_layout()
plt.show()

plot_loss(train_loss_history, test_loss_history)
visualize_filters(model)