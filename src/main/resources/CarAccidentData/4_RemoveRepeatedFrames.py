import pandas as pd
import numpy as np
import os

data = pd.read_csv("OriData/lable_remove.csv", dtype={"DirID":object})

folderToUse = np.unique( pd.read_csv("OriData/lable_remove.csv", dtype={"DirID":object})["DirID"].values )
print(folderToUse)

directory = "OriData/extracted_frames/"

for foldername in folderToUse:
    fullFilePath = directory + foldername

    for curDeleteRange in data[ data["DirID"] == foldername].values:
        print(curDeleteRange)
        for curFileToDelete in range(curDeleteRange[1], curDeleteRange[2]+1):
            curFileToDeletePath = fullFilePath + "/" + str(curFileToDelete) + ".jpg"
            print(curFileToDeletePath)
            os.rename(curFileToDeletePath, curFileToDeletePath.split(".")[0] + ".removedJPG")
            
            

    
