
import cv2
import numpy as np,math

img_1=np.zeros((500,500),dtype=np.uint8)
for i in range (50,301):
    img_1[i,i]=255


Cx,Cy=100,100
r=50
for x in range (Cx-r,Cx+r+1):
    y_square=r**2-(x-Cx)**2
    if y_square >=0:
        y_offest=math.sqrt(y_square)

    y1=int(Cy-y_offest)
    y2=int(Cy+y_offest)
    img_1[y1,x]=255
    img_1[y2,x]=255

cv2.imshow("image1",img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()


img_2=img_1=np.zeros((500,500),dtype=np.uint8)
cv2.line(img_2,(50,50),(300,300),[255,255,255],1)
cv2.circle(img_2,(100,100),50,[255,255,255],1)
cv2.imshow("image2",img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()







