import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import EmotionCNN
# Dataset class for custom CSV loading
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage="Training", transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = np.fromstring(self.data.iloc[idx]['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode='L')
        label = int(self.data.iloc[idx]['emotion'])
        if self.transform:
            img = self.transform(img)
        return img, label

# Prepare transformations for data (convert to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),                  # scales pixel values [0,255] -> [0,1]
    transforms.Normalize((0.5,), (0.5,))    # normalize to [-1,1]
])

# Load Dataset (use 'Training', 'PublicTest', 'PrivateTest' for train/val/test)
full_dataset = FER2013Dataset(r'D:\Codes\FDS\Mini Project\data\fer2013.csv', usage="Training", transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 30

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {epoch_loss:.4f} - Train accuracy: {epoch_acc:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += (preds == labels).sum().item()
            val_total += inputs.size(0)

    val_loss /= val_total
    val_acc = val_corrects / val_total
    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {val_acc:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_emotion_model.pth')
        print("Model improved and saved.")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break


# Face detector from OpenCV
