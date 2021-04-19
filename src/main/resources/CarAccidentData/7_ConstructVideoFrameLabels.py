import cv2
import glob
import numpy as np
import os
import pandas as pd

if not os.path.exists('OriData/videos_labels'):
    os.mkdir('OriData/videos_labels')

labels_df = pd.read_csv('OriData/lable.csv', dtype={'DirID': 'str'})
for video_dir in glob.glob('OriData/videos/*.mp4'):
    cap = cv2.VideoCapture(video_dir)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels = np.zeros(length).astype(int)
    video_name = os.path.basename(video_dir).split('.')[0]
    print(video_dir)
    print(video_name)

    for _, l in labels_df.loc[labels_df.DirID == video_name].iterrows():
        labels[l.Start:l.End+1] = 1

    out_path = f'OriData/videos_labels/{video_name}.csv'
    pd.DataFrame(labels).to_csv(out_path, header=False, index=False)
    with open(out_path) as f:
        lines = f.readlines()
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(out_path, 'w') as wr:
        wr.writelines(lines)