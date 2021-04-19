import os
import glob

dataType = 'ProcessedData'    #'OriData' or 'ProcessedData'

os.makedirs(dataType + '/videos', exist_ok=True)

for filename in glob.glob((dataType + '/extracted_frames2/*/*.mp4')):
    print(dataType + '/videos/' + os.path.basename(filename))
    os.rename(filename, dataType + '/videos/' + os.path.basename(filename))
