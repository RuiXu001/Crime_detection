import numpy as np
import pandas as pd
import cv2
import datetime
import glob
import os
import matplotlib.pyplot as plt
from mongoengine import *
import torch

base_dir = r'D:/research/Iot/video'
os.chdir(base_dir)



def drawBoxes(frame, df):
    boxColor = (128, 255, 0) # very light green
    TextColor = (255, 255, 255) # white
    boxThickness = 1 
    textThickness = 1 
    
    for index, row in df.iterrows():
        xA, yA, xB, yB = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        txt = 'Class: {}  , Conf: {} '.format(row['name'], row['confidence'])
        frame = cv2.rectangle(frame, (xA, yA), (xB, yB), boxColor, boxThickness)
        frame = cv2.putText(frame, txt, (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, textThickness)
    return frame

# load custom model 
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'D:/research/Iot/video/best.pt')

# load pre-trained model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

cap = cv2.VideoCapture('C:/Users/xurui/Downloads/Arson022_x264.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # int width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int height
frames_num = int(cap.get(cv2.CAP_PROP_FPS)) # int fps
total_frame=cap.get(7)
print('Frame per second {}'.format(frames_num))
print('size: {}, {}'.format(width, height))

interval = 1 # collect frame every "interval" second

c = 0
df_col = pd.DataFrame(columns=['frame_num','xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
while True:
    ret, img = cap.read()
    if ret:
        c += 1
        if c % (frames_num * interval) == 0: # detect every 'interval' second
            results = model(img)
            # labels, cords, confids = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-2].numpy(), results.xyxyn[0][:, -2].numpy()
            df = results.pandas().xyxy[0]
            if len(df) > 0: # at least one object detected
                df['frame_num'] = c
                df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
                df_col = pd.concat([df_col, df],ignore_index=True)# for ind in range(len(df)):
                print(datetime.datetime.now(), ' --- frame ', c, len(df_col))
        cv2.namedWindow("video", 0)
        cv2.resizeWindow("video", width*3, height*3)
        cv2.imshow('video',drawBoxes(img,df))
        k = cv2.waitKey(10) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

df_col.to_csv('')

# object and count
df_col.groupby(by = 'name').count()['frame_num']
# object detected for each frame
df_col.groupby(by = 'frame_num').count()['name']
