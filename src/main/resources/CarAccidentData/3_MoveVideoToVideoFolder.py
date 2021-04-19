import os
import glob

os.makedirs('OriData/videos', exist_ok=True)

for filename in glob.glob(f'OriData/extracted_frames/*/*.mp4'):
    print('OriData/videos/' + os.path.basename(filename))
    os.rename(filename, 'OriData/videos/' + os.path.basename(filename))
