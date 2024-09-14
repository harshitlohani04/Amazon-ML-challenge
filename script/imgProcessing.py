import cv2 as cv
import PIL
from PIL import Image
from entity import allowed_units
import pytesseract as pyts
import os
import matplotlib.pyplot as plt

# Initiallizing the root directory
rd = "D:/Amazon-ML-challenge/script/"

pyts.pytesseract.tesseract_cmd = r"D:/tesseract/tesseract.exe"

try:
    img = Image.open(os.path.join(rd, "sampleData/image1.jpg"))
    extrText = pyts.image_to_string(img)
    print(extrText)
except PIL.UnidentifiedImageError as e:
    print("Image was not identified")
except FileNotFoundError as e:
    print("The image was not found")

