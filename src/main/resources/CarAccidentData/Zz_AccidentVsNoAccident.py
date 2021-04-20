import pandas as pd
import numpy as np
from natsort import natsorted, ns
import os
import shutil

inLabelDir = "OriData/videos_labels/"


labelFiles = [lbl for lbl in os.listdir(inLabelDir) if lbl.endswith(".csv")]
labelFiles = natsorted(labelFiles)
print(labelFiles)


priorColLength = 90
afterColLength = 20

curAindex = 0
curNAindex = 0

for i, fileName in enumerate(labelFiles):
    data = pd.read_csv(inLabelDir + fileName, header=None)
    
    startingFrameIndex = None
    endingFrameIndex = None

    allCollisionFrameIndex = data[data[0] == 1].index.values
    print(allCollisionFrameIndex)

    if data[data[0] == 1].index[0] <= priorColLength:
        startingFrameIndex = 0
    else:
        startingFrameIndex = data[data[0] == 1].index[0] - priorColLength

    endingFrameIndex = startingFrameIndex + priorColLength + afterColLength

    os.makedirs("AvNA/A/", exist_ok=True)
    os.makedirs("AvNA/NA/", exist_ok=True)

    
        
    for curIndex in range( startingFrameIndex, endingFrameIndex + 1 ):
        if os.path.exists("OriData/extracted_frames/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg'):
            if curIndex in allCollisionFrameIndex:
                shutil.copyfile( "OriData/extracted_frames/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg',
                                 "AvNA/A/" + str(curAindex) + '.jpg')
                curAindex += 1
            else:
                shutil.copyfile( "OriData/extracted_frames/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg',
                                 "AvNA/NA/" + str(curNAindex) + '.jpg')
                curNAindex += 1
                
