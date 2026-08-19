import cv2
import numpy as np 
import matplotlib.pyplot as plt 


COL_BOX=(0,255,0)
COL_CENTROID=(0,0,255)
COL_OVERLAY=(0,255,255)
img=np.zeros((700,1200,3),dtype=np.uint8)
cv2.rectangle(img,(50,50),(250,250),COL_BOX,3)
cv2.circle(img,(400,150),50,COL_CENTROID,-1)
cv2.line(img,(50,300),(500,300),COL_OVERLAY,5)
cv2.ellipse(img,(800,150),(150,70),0,0,360,COL_BOX,3)
cv2.arrowedLine(img,(650,350),(950,350),COL_CENTROID,5,tipLength=0.15)
pts=np.array([[700,450],[850,400],[1000,450],[950,600],[750,600]],np.int32)
cv2.polylines(img,[pts],True,COL_OVERLAY,3)
cv2.fillPoly(img,[pts],COL_CENTROID)
text="OpenCV Drawing"
x=450
y=50
padding=5
(w,h),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.5,3)
cv2.rectangle(img,(x-padding,y-h-padding),(x+w+padding,y+padding+baseline),COL_BOX,2)
cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
overlay=img.copy()
cv2.rectangle(overlay,(400,400),(600,500),COL_BOX,-1)
img=cv2.addWeighted(img,0.7,overlay,0.3,0)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


























