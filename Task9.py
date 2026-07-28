import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-27 at 14.08.27.jpeg")
if img is None:
    raise FileNotFoundError("The img not loaled")


img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Threshold_Value=70
T,img_Binary=cv2.threshold(img_gray,Threshold_Value,255,cv2.THRESH_BINARY)
T,img_BinaryInv=cv2.threshold(img_gray,Threshold_Value,255,cv2.THRESH_BINARY_INV)
T,img_Trunc=cv2.threshold(img_gray,Threshold_Value,255,cv2.THRESH_TRUNC)
T,img_ToZero=cv2.threshold(img_gray,Threshold_Value,255,cv2.THRESH_TOZERO)
T,img_ToZeroInv=cv2.threshold(img_gray,Threshold_Value,255,cv2.THRESH_TOZERO_INV)

T_Otsu,img_BinaryOtsu=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_AdaptiveGaussian=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=11,C=2)

fig,axes=plt.subplots(2,4,figsize=(14,8))
axes[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0,0].set_title("img_Orig")
axes[0,0].axis("off")
axes[0,1].imshow(img_Binary,cmap='gray')
axes[0,1].set_title("img_Binary")
axes[0,1].axis("off")
axes[0,2].imshow(img_BinaryInv,cmap='gray')
axes[0,2].set_title("img_BinaryInv")
axes[0,2].axis("off")
axes[0,3].imshow(img_Trunc,cmap='gray')
axes[0,3].set_title("img_Trunc")
axes[0,3].axis("off")
axes[1,0].imshow(img_ToZero,cmap='gray')
axes[1,0].set_title("img_ToZero")
axes[1,0].axis("off")
axes[1,1].imshow(img_ToZeroInv,cmap='gray')
axes[1,1].set_title("img_ToZeroInv")
axes[1,1].axis("off")
axes[1,2].imshow(img_BinaryOtsu,cmap='gray')
axes[1,2].set_title("img_BinaryOtsu")
axes[1,2].axis("off")
axes[1,3].imshow(img_AdaptiveGaussian,cmap='gray')
axes[1,3].set_title("img_AdaptiveGaussian")
axes[1,3].axis("off")
plt.tight_layout()
plt.show()
