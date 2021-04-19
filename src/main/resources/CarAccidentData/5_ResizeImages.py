import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("opencv-python")
install("natsort")
install("pandas")
install("pillow")






imageWidth = 224
imageHeight = 224

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import pandas as pd
import numpy as np
from natsort import natsorted, ns


folderToUse = np.unique( pd.read_csv("OriData/lable.csv", dtype={"DirID":object})["DirID"].values )
print(folderToUse)



for foldername in folderToUse:
    inFullFolderPath = "OriData/extracted_frames/" + foldername
    outFullFolderPath = "ProcessedData/extracted_frames/" + foldername
    os.makedirs(outFullFolderPath, exist_ok=True)
    print(inFullFolderPath)
    print(outFullFolderPath)

    inImages = [img for img in os.listdir(inFullFolderPath) if img.endswith(".jpg")]
    inImages = natsorted(inImages)
    print(inImages)

    for oneInImage in inImages:
        img = Image.open(inFullFolderPath + "/" + oneInImage)

        img = img.resize((imageWidth, imageHeight), Image.ANTIALIAS)   # Squeeze Image
        img.save(outFullFolderPath + "/" + oneInImage)
        

