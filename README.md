# Traffic Accident Detection with DL4J

Team Member:
- Fletcher Sarip
- David Ooi

## Dataset
[CADP dataset](https://ankitshah009.github.io/accident_forecasting_traffic_camera) is used to train the deep learning model to detect traffic accident in CCTV video. We manually labeled the extracted image frames with label 0 for "no accident" and 1 for "accident". Only video frames corresponding to collision and "after-effects" of collisions are considered labeled as "accident". If the after-effect of the collision already settled down, the video frames are back to being labeled as "no accident".

## Deep Learning Model
CNN + LSTM model is implemented using DL4J framework to detect traffic accidents in CCTV video frames.
The CCTV video sequence classification is done in "many-to-many" fashion, i.e. every frame of the CCTV video is classified as either "accident" or "no accident" based on the current video frame and previously seen video frames.
