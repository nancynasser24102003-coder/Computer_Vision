import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

Blur_K=(5,5)
Canny_Low=50
Canny_High=150
Edges_Color=(0,255,0)
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Can not Open Camera")

while True :

    ret,frame=cap.read()

    if not ret :
        break

    Blurred=cv2.GaussianBlur(frame,Blur_K,0)
    Gray_Blurred=cv2.cvtColor(Blurred,cv2.COLOR_BGR2GRAY)
    Edges=cv2.Canny(Gray_Blurred,Canny_Low,Canny_High,apertureSize=5,L2gradient=True)
    output=frame.copy()
    output[Edges==255]=Edges_Color
    Edges_BGR=cv2.cvtColor(Edges,cv2.COLOR_GRAY2BGR)
    Combined=np.hstack([frame,Edges_BGR,output])
    Combined=cv2.resize(Combined,(1500,500))
    cv2.imshow("Edges Layer",Combined)


    if cv2.waitKey(1) & 0xFF==ord('c'):
      break
    if 0xFF==ord('c'):
        cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\Video_Capture.png",Combined)

cap.release()
cv2.destroyAllWindows()
    
















