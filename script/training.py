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

rd = "D:/Amazon-ML-challenge/student_resource/"
trainData = pd.read_csv(os.path.join(rd, "dataset/train.csv"))[:5]


def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return
    

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)
    
    create_placeholder_image(image_save_path) #Create a black placeholder image for invalid links/images


def train_crnn(model, train_loader, num_epochs=10):
    criterion = nn.CTCLoss(blank=0)  # blank token for CTC
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets, target_lengths in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # Outputs shape: (sequence_length, batch_size, num_classes)

            # Compute CTC loss
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)  # Sequence length
            loss = criterion(outputs, targets, input_lengths, target_lengths)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')


print(trainData)
# Example of loading dataset and training
# from torch.utils.data import DataLoader
# # Assuming your custom dataset that loads images and their labels
# train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
# model = CRNN(num_classes=NUM_CLASSES)
# train_crnn(model, train_loader)
