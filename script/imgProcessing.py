import cv2 as cv
import PIL
from PIL import Image
from entity import allowed_units
import pytesseract as pyts
import os
import sys
import matplotlib.pyplot as plt

# Preprocessing Function
def preprocessing(image_path):
    img = cv.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {img_path}. Please check the file path and integrity.")
        return None
    
    img = cv.resize(img, (1000, 1000))
    img = img[200:800, 200:800]
    cv.imshow(' ',img)
    cv.waitKey(0)
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    unNoisedImg = cv.GaussianBlur(grayImg, (3, 3), 0) # Attempting Noise removal using Gaussian Blur
    # remNoiseimg = cv.medianBlur(grayImg, 3)

    try:
        extrText = pyts.image_to_string(unNoisedImg)
        return extrText
    except pyts.pytesseract.TesseractNotFoundError as e:
        print("Tesseract location not found")
        return -1


if __name__ == "__main__":
    # Initiallizing the root directory
    rd = "D:/Amazon-ML-challenge/script/"

    pyts.pytesseract.tesseract_cmd = r"D:/tesseract/tesseract.exe"

    try:
        img_path = os.path.join(rd, "sampleData/sample.jpg")
        extText = preprocessing(img_path)
        if extText == -1:
            print("There was an error in tesseract")
        else:
            print(extText)
    except PIL.UnidentifiedImageError as e:
        print("Image was not identified")
        sys.exit(0)
    except FileNotFoundError as e:
        print("The image was not found")
        sys.exit(0)

