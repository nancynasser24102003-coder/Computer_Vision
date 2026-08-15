import cv2
import numpy as np
import matplotlib.pyplot as plt

Lower_Range=np.array([35,70,50])
Upper_Range=np.array([85,255,255])
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("The Camera can not open")

while True:
    ret,frame=cap.read()
    if not ret :
        break
    frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(frame_hsv,Lower_Range,Upper_Range)
    Kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,Kernel)
    clean_mask=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,Kernel)
    Contours,Hierarchy=cv2.findContours(clean_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Min_Area=500
    Valid_Contours=[]
    for c in Contours :
        Area=cv2.contourArea(c)
        if Area>=Min_Area:
            Valid_Contours.append(c)
    if len(Valid_Contours)>0:
      Largest_Contour=max(Valid_Contours,key=cv2.contourArea)
      Largest_Area=cv2.contourArea(Largest_Contour)
      x,y,w,h=cv2.boundingRect(Largest_Contour)
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
      cv2.putText(frame,"Box",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
      M=cv2.moments(Largest_Contour)
      if M["m00"]!=0:
          Cx=int(M["m10"]/M["m00"])
          Cy=int(M["m01"]/M["m00"])
      cv2.circle(frame,(Cx,Cy),5,(0,0,255),-1)
      cv2.putText(frame,"Centroid",(Cx+10,Cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
      Aspect_Ratio=w/h
      print(f"Area:",Area)
      print(f"Bounding Rect (x,y,w,h):",x,y,w,h)
      print(f"Centroid (Cx,Cy):",Cx,Cy)
      print(f"Aspect_Ratio:",Aspect_Ratio)
    cv2.imshow("Camera",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('s'):
        cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\frame.png",frame)
        cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\mask.png",clean_mask)

    if key==ord('c'):
        break
cap.release()
cv2.destroyAllWindows()



















































