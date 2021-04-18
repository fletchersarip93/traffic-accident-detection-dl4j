import pandas as pd
import numpy as np
import shutil
import os

data = pd.read_csv("OriData/lable.csv", dtype={"DirID":object})

allSelectedFolder = np.unique( data['DirID'].values )

for oneFolder in allSelectedFolder:
  allAccidentRange = data[ data['DirID'] == oneFolder].values

  for oneAccidentRange in allAccidentRange:
    for x in range( oneAccidentRange[1], oneAccidentRange[2] + 1 ):
      src = "OriData/extracted_frames/" + oneFolder + "/" + str(x) + ".jpg"
      des = "ProcessedData/extracted_frames/" + oneFolder + "/" + str(x) + ".jpg"
      os.makedirs(os.path.dirname(des), exist_ok=True)
      shutil.copyfile(src, des)
  print("___")
