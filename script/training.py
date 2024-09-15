import torch.optim as optim
import torch.nn as nn
import torch
from model import CRNN
import pandas as pd
import os
from PIL import Image
import time
from pathlib import Path
import urllib
from dataloader import ProductDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


rd = "student_resource"
trainData = pd.read_csv(os.path.join(rd, "dataset/train.csv"))[:5]

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    
    # Stack images (assuming images are tensors of shape [C, H, W])
    images = torch.stack(images, 0)
    
    # Pad targets and create tensors for target_lengths
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # Padding value 0 is assumed
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return images, targets, target_lengths


def train_crnn(model, train_loader, num_epochs=10):
    criterion = nn.CTCLoss()  # blank token for CTC
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets, target_lengths in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # Outputs shape: (sequence_length, batch_size, num_classes)
            
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
            input_lengths = torch.full(size=(outputs.size(0),), 
                                       fill_value=outputs.size(1), 
                                       dtype=torch.long)


            # Compute CTC loss
            # input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)  # Sequence length
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Example of loading dataset and training

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

rd = "student_resource"

character_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

train_df = pd.read_csv(os.path.join(rd, "dataset/train.csv"))[:5]
train_dataset = ProductDataset(data=train_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = CRNN(num_classes=len(character_map), input_channels=1)
train_crnn(model, train_loader)

