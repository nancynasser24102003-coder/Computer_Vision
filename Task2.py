import cv2
import numpy as np ,math
img=np.zeros((500,500),dtype=np.uint8)



for x in range (0,499):
    y=x+1
    img[y,x]=255



r=50
Cx,Cy=350,150

for X in range (Cx-r,Cx+r+1):
    Y_Square=r**2-(X-Cx)**2
    if  Y_Square >=0:
      Y_Offset=math.sqrt(Y_Square)
      Y1=int(Cy+Y_Offset)
      Y2=int(Cy-Y_Offset)
      img[X,Y1]=255
      img[X,Y2]=255
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



