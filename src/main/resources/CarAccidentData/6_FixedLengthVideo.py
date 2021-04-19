import pandas as pd
import numpy as np
from natsort import natsorted, ns
import os
import shutil

inLabelDir = "OriData/videos_labels/"

labelFiles = [lbl for lbl in os.listdir(inLabelDir) if lbl.endswith(".csv")]
labelFiles = natsorted(labelFiles)
print(labelFiles)


priorColLength = 180  # The frames before the first collision frame
afterColLength = 180  # The frames after the first collision frame
totalFramesMax = priorColLength + afterColLength + 1
# So total you have 361 frames


for i, fileName in enumerate(labelFiles):
    data = pd.read_csv(inLabelDir + fileName, header=None)

    print( data.index[-1] )
    #print( data[data[0] == 1].index[0] )
    #print( len( data[data[0] == 1] ) )
    #print(len(data))
    

    startingFrameIndex = None
    endingFrameIndex = None

    if data[data[0] == 1].index[0] <= priorColLength:
        startingFrameIndex = 0
    else:
        startingFrameIndex = data[data[0] == 1].index[0] - priorColLength

    endingFrameIndex = startingFrameIndex + priorColLength + afterColLength
    '''
    totalFramesNow = data.index[-1] - startingFrameIndex + 1
    if totalFramesNow < totalFramesMax:
        endingFrameIndex = startingFrameIndex + priorColLength + afterColLength
        AddBlankFrames = True
    else:
        endingFrameIndex = startingFrameIndex + priorColLength + afterColLength
    '''

    os.makedirs("ProcessedData/extracted_frames2/" + fileName.split(".")[0], exist_ok=True)


    for curIndex in range( startingFrameIndex, endingFrameIndex + 1 ):
        #print(curIndex) #Need to create this frame
        if os.path.exists("ProcessedData/extracted_frames/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg'):
            shutil.copyfile( "ProcessedData/extracted_frames/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg',
                             "ProcessedData/extracted_frames2/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg')
        else:
            shutil.copyfile( "_Blank.jpg",
                             "ProcessedData/extracted_frames2/" + fileName.split(".")[0] + "/" + str(curIndex) + '.jpg')


    print( "Start", startingFrameIndex )
    print( "END", endingFrameIndex )
    lableDataReady = data[startingFrameIndex:].values
    lableString = ""
    for x in range(0, totalFramesMax):
        try:
            lableString += str(lableDataReady[x][0]) + "\n"
        except:
            lableString += "0\n"
    #print(lableString)

    os.makedirs("ProcessedData/videos_labels2", exist_ok=True)
    
    with open("ProcessedData/videos_labels2/" + fileName, "w") as outputCSV:
        outputCSV.write( lableString )

