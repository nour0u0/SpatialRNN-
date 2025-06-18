import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Config
config = {
    "patch_size": 6,
    "stride": 3,  # Overlap: stride < patch_size
    "hidden_state_dim": 128,
    "batch_size": 64,
    "epochs": 1,
    "learning_rate": 5e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "img_size": 32,
    "channels": 3,
}

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)

valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=config["batch_size"], shuffle=False)

# Overlapping patch extraction
def split_patches_overlap(imgs, patch_size, stride):
    batch_size, channels, H, W = imgs.shape
    patches = imgs.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    patches = patches.view(batch_size, -1, channels * patch_size * patch_size)
    return patches

# Overlapping patch reconstruction (averaging overlaps)
def patches_to_img_overlap(patches, patch_size, stride, img_size, channels):
    batch_size, n_patches, patch_len = patches.shape
    patches_per_row = (img_size - patch_size) // stride + 1

    patches = patches.view(batch_size, n_patches, channels, patch_size, patch_size)

    recon_img = torch.zeros(batch_size, channels, img_size, img_size, device=patches.device)
    weight = torch.zeros_like(recon_img)

    patch_idx = 0
    for i in range(patches_per_row):
        for j in range(patches_per_row):
            recon_img[:, :, i*stride:i*stride+patch_size, j*stride:j*stride+patch_size] += patches[:, patch_idx]
            weight[:, :, i*stride:i*stride+patch_size, j*stride:j*stride+patch_size] += 1
            patch_idx += 1

    return recon_img / weight

# Model definition: MLP patch model with hidden state
class SpatialRouterWithHiddenState(nn.Module):
    def __init__(self, patch_size, hidden_state_dim, channels=3):
        super().__init__()
        input_dim = patch_size * patch_size * channels
        hidden_dim = 256
        self.fc1 = nn.Linear(input_dim + hidden_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim + hidden_state_dim)
        self.relu = nn.ReLU()

    def forward(self, x, hidden_state):
        combined = torch.cat([x, hidden_state], dim=1)
        h = self.relu(self.fc1(combined))
        out = self.fc2(h)
        pred_patch = out[:, :x.shape[1]]
        next_hidden = out[:, x.shape[1]:]
        return pred_patch, next_hidden

# Visualization of reconstruction results
def visualize_results(clean_imgs, reconstructed_imgs):
    batch_size = clean_imgs.shape[0]
    fig, axs = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))
    for i in range(batch_size):
        axs[0, i].imshow(np.transpose(clean_imgs[i].cpu().numpy(), (1, 2, 0)))
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title('Original')

        axs[1, i].imshow(np.transpose(reconstructed_imgs[i].cpu().numpy(), (1, 2, 0)))
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title('Reconstructed')
    plt.tight_layout()
    plt.show()

# Run one epoch (train or val)
def run_epoch(model, dataloader, optimizer=None, train=True):
    device = config["device"]
    criterion = nn.MSELoss()
    losses = []

    if train:
        model.train()
    else:
        model.eval()

    for imgs, _ in tqdm(dataloader, desc="Training" if train else "Validation"):
        imgs = imgs.to(device)
        input_patches = split_patches_overlap(imgs, config["patch_size"], config["stride"])

        batch_size, n_patches, patch_len = input_patches.shape
        hidden_state = torch.zeros(batch_size, config["hidden_state_dim"], device=device)

        if train:
            optimizer.zero_grad()

        pred_patches = []
        for i in range(n_patches):
            patch_in = input_patches[:, i, :]
            pred_patch, hidden_state = model(patch_in, hidden_state)
            pred_patches.append(pred_patch)

        pred_patches = torch.stack(pred_patches, dim=1)
        loss = criterion(pred_patches, input_patches)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    return losses

def train_model():
    device = config["device"]
    model = SpatialRouterWithHiddenState(config["patch_size"], config["hidden_state_dim"], config["channels"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        batch_train_losses = run_epoch(model, trainloader, optimizer, train=True)
        batch_val_losses = run_epoch(model, valloader, optimizer=None, train=False)

        train_losses.extend(batch_train_losses)
        val_losses.extend(batch_val_losses)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {current_lr:.6f} - Train Loss (last batch): {batch_train_losses[-1]:.6f} - Val Loss (last batch): {batch_val_losses[-1]:.6f}")

    # Plot raw loss values per batch (train and val)
    plt.figure(figsize=(12,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Batch (Raw Values)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization after training
    model.eval()
    with torch.no_grad():
        test_imgs = next(iter(valloader))[0].to(device)[:8]
        input_patches = split_patches_overlap(test_imgs, config["patch_size"], config["stride"])
        batch_size, n_patches, patch_len = input_patches.shape
        hidden_state = torch.zeros(batch_size, config["hidden_state_dim"], device=device)
        pred_patches = []
        for i in range(n_patches):
            patch_in = input_patches[:, i, :]
            pred_patch, hidden_state = model(patch_in, hidden_state)
            pred_patches.append(pred_patch)
        pred_patches = torch.stack(pred_patches, dim=1)
        reconstructed_imgs = patches_to_img_overlap(pred_patches, config["patch_size"], config["stride"], config["img_size"], config["channels"])
        visualize_results(test_imgs[:8].cpu(), reconstructed_imgs[:8].cpu())

if __name__ == "__main__":
    train_model()
