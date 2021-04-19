from PIL import Image
import os
import pandas as pd
import numpy as np

folderToUse = np.unique( pd.read_csv("OriData/lable.csv", dtype={"DirID":object})["DirID"].values )
print(folderToUse)

directory = "OriData/extracted_frames/"

for foldername in folderToUse:
#for foldername in os.listdir(directory):
        folderN = os.path.join(directory, foldername)

        for filename in os.listdir(folderN):
                f = os.path.join(folderN, filename)
                 
                if os.path.isfile(f):
                        try:
                                if filename.split(".")[-1] == "jpg":
                                        im = Image.open(f)
                                else:
                                        DoNothing = True #print("redundant " + foldername + "/" + filename)
                        except:
                                print(foldername + "/" + filename)
                                #print("CORRUPTED " + foldername + "/" + filename)
                else:
                        print("UNKNOWN " + filename)
