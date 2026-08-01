import cv2
import numpy as np 
import matplotlib.pyplot as plt 


Lower_YCrCb=np.array([0,133,77])
Upper_YCrCb=np.array([255,173,127])
Kerne_Elipse5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cap=cv2.VideoCapture(0)


if not cap.isOpened():
    raise RuntimeError("Can not Open Cammera")

while True:
    ret,frame=cap.read()
    if not ret:
        break

    Frame_Flip=cv2.flip(frame,1)
    Frame_LAB=cv2.cvtColor(Frame_Flip,cv2.COLOR_BGR2LAB)
    L,A,B=cv2.split(Frame_LAB)
    Clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(5,5))
    L_enh=Clahe.apply(L)
    Frame_enh=cv2.merge([L_enh,A,B])
    Frame_enh_BGR=cv2.cvtColor(Frame_enh,cv2.COLOR_LAB2BGR)
    Frame_YCrCb=cv2.cvtColor(Frame_enh_BGR,cv2.COLOR_BGR2YCrCb)
    Mask=cv2.inRange(Frame_YCrCb,Lower_YCrCb,Upper_YCrCb)
    Mask_Clean_Open=cv2.morphologyEx(Mask,cv2.MORPH_OPEN,Kerne_Elipse5)
    Mask_Clean=cv2.morphologyEx(Mask_Clean_Open,cv2.MORPH_CLOSE,Kerne_Elipse5)
    
    Frame_Result=cv2.bitwise_and(Frame_enh_BGR,Frame_enh_BGR,mask=Mask)
    Frame_Result_gray=cv2.cvtColor(Frame_Result,cv2.COLOR_BGR2GRAY)
    Edges=cv2.Canny(Frame_Result_gray,50,150,apertureSize=3,L2gradient=True)
    Output=Frame_Result.copy()
    Output[Edges==255]=(0,255,0)
    cv2.imshow("Face",Output)





    








    key=cv2.waitKey(1) & 0xFF
    if key ==ord('c'):
      break
    if key==ord('s'):
        cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\Proget_Image.png")




cap.release()
cv2.destroyAllWindows()



