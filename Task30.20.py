import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img_BackGround=cv2.imread(r"C:\Users\Etijah\Pictures\WhatsApp Image 2026-08-02 at 15.00.44.jpeg")
img_Logo=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-08-02 at 15.00.44.jpeg")
if img_BackGround is None or img_Logo is None :
    raise FileNotFoundError("The Image not Loaded")

H_BackGround,W_BackGround=img_BackGround.shape[:2]

H_Logo=int(H_BackGround*0.15)

W_Logo=int(W_BackGround*0.15)

img_Logo=cv2.resize(img_Logo,(W_Logo,H_Logo),cv2.INTER_AREA)

img_Logo_Gray=cv2.cvtColor(img_Logo,cv2.COLOR_BGR2GRAY)

T,Mask=cv2.threshold(img_Logo_Gray,10,255,cv2.THRESH_BINARY)

Mask_Inv=cv2.bitwise_not(Mask)

ROI_BackGround=img_BackGround[0:H_Logo,0:W_Logo]

BG=cv2.bitwise_and(ROI_BackGround,ROI_BackGround,mask=Mask_Inv)

FG=cv2.bitwise_and(img_Logo,img_Logo,mask=Mask)

Roi_Final=cv2.add(BG,FG)
img_Final=img_BackGround.copy()
img_Final[0:H_Logo,0:W_Logo]=Roi_Final

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_BackGround, cv2.COLOR_BGR2RGB))
plt.title("Background")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_Logo, cv2.COLOR_BGR2RGB))
plt.title("Logo")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_Final, cv2.COLOR_BGR2RGB))
plt.title("Final")
plt.axis("off")

plt.tight_layout()
plt.show()


















