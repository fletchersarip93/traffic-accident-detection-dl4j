import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("opencv-python")
install("natsort")
install("pandas")



import os
import pandas as pd
import numpy as np
from natsort import natsorted, ns
import cv2

folderToUse = np.unique( pd.read_csv("OriData/lable.csv", dtype={"DirID":object})["DirID"].values )
print(folderToUse)

directory = "OriData/extracted_frames/"

videoFPS = 30

for foldername in folderToUse:
#for foldername in os.listdir(directory):
    fullFolderPath = directory + foldername
    print(fullFolderPath)

    images = [img for img in os.listdir(fullFolderPath) if img.endswith(".jpg")]
    images = natsorted(images)
    print(images)

    frame = cv2.imread(os.path.join(fullFolderPath, images[0]))
    height, width, layers = frame.shape

    curSaveVideoPath = fullFolderPath + "/" + foldername + ".mp4"
    print(curSaveVideoPath)
    video = cv2.VideoWriter(curSaveVideoPath, 0, videoFPS, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(fullFolderPath, image)))

    cv2.destroyAllWindows()
    video.release()

'''
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("opencv-python")
install("natsort")


from natsort import natsorted, ns
import cv2
import os

image_folder = "OriData/extracted_frames/000000"
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
'''
