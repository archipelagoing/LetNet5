# 9.20pm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io

class ParquetDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Decode image (from bytes)
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Convert label to tensor
        label = torch.tensor(row['label'], dtype=torch.long)

        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create dataset and dataloader
dataset = ParquetDataset("path_to_parquet_file.parquet", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example: Iterate through DataLoader
for images, labels in dataloader:
    print(f"Batch of images shape: {images.shape}")  # [16, 1, 32, 32]
    print(f"Batch of labels: {labels}")             # [16]
    break
