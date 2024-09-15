import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
import os
import pandas as pd


import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO

class ProductDataset(Dataset):
    def __init__(self, data, transform=None, char_to_idx=None):
        """
        Args:
            data (pd.DataFrame): A DataFrame containing image links, entity names, and entity values.
            transform (callable, optional): Optional transform to be applied on a sample.
            char_to_idx (dict, optional): Dictionary to map characters to indices for the entity_value.
        """
        self.data = data
        self.transform = transform
        self.char_to_idx = char_to_idx or self.create_char_to_idx()

    def create_char_to_idx(self):
        """Create a default character-to-index mapping."""
        import string
        char_set = string.ascii_letters + string.digits + ' '  # Includes letters, digits, and space
        return {char: idx for idx, char in enumerate(char_set, start=1)}  # Start index from 1

    def encode_value(self, entity_value):
        """Encodes the entity_value (a string) into a list of indices."""
        return [self.char_to_idx[char] for char in entity_value if char in self.char_to_idx]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image from URL
        img_url = self.data.iloc[idx]['image_link']
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        # Encode the entity_value (target) as a sequence of character indices
        entity_value = self.data.iloc[idx]['entity_value']
        target = self.encode_value(entity_value)
        target_length = len(target)

        return img, torch.tensor(target, dtype=torch.long), target_length


# Define image transformations (resize, convert to tensor, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


