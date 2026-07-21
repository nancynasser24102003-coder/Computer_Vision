import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-21 at 16.44.45.jpeg")
if img is None:
    raise FileNotFoundError("error")
else:
    print(img.shape)

img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_range=np.array([0,100,100])
upper_range=np.array([10,255,255])
mask=cv2.inRange(img_HSV,lower_range,upper_range)
mask_inv=cv2.bitwise_not(mask)
result=cv2.bitwise_and(img,img,mask=mask) 
cv2.imshow("Original_img",img)
cv2.imshow("Mask",mask)
cv2.imshow("segmented_img",result)
cv2.imshow("Mask_not",mask_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(img_LAB)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
L_enh=clahe.apply(L)
# L_enh=cv2.add(L,50) instead of clahe 
img_enh=cv2.merge([L_enh,A,B])
img_back=cv2.cvtColor(img_enh,cv2.COLOR_LAB2BGR)
cv2.imshow("Original_img",img)
cv2.imshow("Edited_img",img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
foreground=cv2.bitwise_and(img_back,img_back,mask=mask)
background=cv2.bitwise_and(gray_bgr,gray_bgr,mask=mask_inv) 
result_img=cv2.add(background,foreground)
cv2.imshow("Final",result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()