import torch
import torch.nn as nn
from training import model
import os
import urllib
from pathlib import Path
from PIL import Image
import time
import cv2
from postProcessing import extract_entity_from_text

import torch

def decode_ctc_output(output, character_map):
    """
    Decode the output of the CRNN model using CTC decoding.

    Parameters:
    - output: The raw output from the CRNN model (shape: (sequence_length, batch_size, num_classes)).
    - character_map: A list mapping class indices to characters.

    Returns:
    - decoded_text: The decoded text from the output probabilities.
    """
    # Convert the output probabilities to character indices
    character_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
    output = output.permute(1, 0, 2)  # (batch_size, sequence_length, num_classes)
    output = torch.argmax(output, dim=2)  # Get the indices of the max probability

    decoded_text = []
    for i in range(output.size(0)):  # Loop over each image in the batch
        text = []
        prev_char = None
        for char_index in output[i]:
            char = character_map[char_index.item()]
            if char != '' and char != prev_char:  # Remove repeated characters and blanks
                text.append(char)
            prev_char = char
        decoded_text.append(''.join(text))

    return decoded_text


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 32))  # Resize to standard input size for model
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarization
    img = cv2.medianBlur(img_bin, 3)  # Apply noise reduction
    return img


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

def inference(image_url, entity_name):
    img = download_image(image_url)
    processed_img = preprocess_image(img)
    processed_img = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).float()  # Prepare for model input

    model.eval()
    with torch.no_grad():
        output = model(processed_img)
        predicted_text = decode_ctc_output(output)  # You need to implement this function

    entity_value = extract_entity_from_text(predicted_text, entity_name)
    return entity_value
