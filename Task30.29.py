import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
if img is None :
    raise FileNotFoundError("The Image not Loaded")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
H,W=img.shape[:2]
img_black=np.zeros((H,W),dtype=np.uint8)
Cx,Cy=W//2,H//2
R=min(H,W)//3
mask=cv2.circle(img_black,(Cx,Cy),R,255,thickness=-1)
hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
hist_mask=cv2.calcHist([img_gray],[0],mask,[256],[0,256])
img_mask=cv2.bitwise_and(img,img,mask=mask)


plt.plot(hist.flatten(),color='red',label='Histogram of Img')
plt.plot(hist_mask.flatten(),color='blue',label='Histogram of Mask')
plt.title("Full Image vs Masked Region Histogram")
plt.legend()
plt.xlabel("Intensity")
plt.ylabel("Num of Pixels")
plt.xlim([0,256])
plt.show()



plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Original Image ")
plt.subplot(1,3,2)
plt.imshow(mask,cmap='gray')
plt.axis("off")
plt.title(" Mask ")
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(img_mask,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(" Masked Image  ")
plt.tight_layout()
plt.show()











