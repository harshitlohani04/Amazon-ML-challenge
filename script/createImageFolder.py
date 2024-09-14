import urllib
import os
from pathlib import Path
import time
from PIL import Image
import pandas as pd


rd = "./student_resource/dataset/"

'''
    These 2 functions have been taken from utils.py in the student resource.

    create_placeholder_image(image_save_path) --> This is for creating a black image inplace if
                                                  the image link fails to open up.

    download_image(image_link, save_folder, retries=3, delay=3) --> TO download the image.
'''

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return
    

# Image downloading function
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


trainData = pd.read_csv(os.path.join(rd, "train.csv"))



