import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import time 
from collections import deque

FPS_Time=deque(maxlen=30)
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("The Camera Can not Open")
while True:
    
    ret,frame=cap.read()
    if not ret :
        break
    start_time=time.perf_counter()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_blurred=cv2.GaussianBlur(frame_gray,(5,5),0)
    end_time=time.perf_counter()
    elapsed=end_time-start_time
    FPS_Time.append(elapsed)
    Average_Time=sum(FPS_Time)/len(FPS_Time)
    FPS=1/ Average_Time
    data=[f"FPS {FPS:.1f}","Objects","Area:1024 px"]
    x=20
    y=40
    gap=40
    for i ,text in enumerate(data):
        cv2.putText(frame_blurred,text,(x,y+i*gap),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("camera",frame_blurred)
    if cv2.waitKey(1) & 0xFF==ord('c'):
        break
cap.release()
cv2.destroyAllWindows() 




