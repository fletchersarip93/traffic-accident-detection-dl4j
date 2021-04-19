import shutil
import os
from natsort import natsorted, ns

inVideoDir = "ProcessedData/videos/"
inLabelDir = "ProcessedData/videos_labels2/"

videoFiles = [vid for vid in os.listdir(inVideoDir) if vid.endswith(".mp4")]
videoFiles = natsorted(videoFiles)
print(videoFiles)

labelFiles = [lbl for lbl in os.listdir(inLabelDir) if lbl.endswith(".csv")]
labelFiles = natsorted(labelFiles)
print(labelFiles)

for i, fileName in enumerate(videoFiles):
    shutil.copyfile( inVideoDir + fileName, "zProperData/" + str(i) + ".mp4")

for i, fileName in enumerate(labelFiles):
    shutil.copyfile( inLabelDir + fileName, "zProperData/" + str(i) + ".csv")

