import cv2
import numpy as np 
import matplotlib.pyplot as plt


img_gray=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-28 at 17.12.50.jpeg",cv2.IMREAD_GRAYSCALE)
T,mask=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
Kernel_Rect5=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
Kernel_Elipse15=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
mask_Opened_Rect=cv2.morphologyEx(mask,cv2.MORPH_OPEN,Kernel_Rect5,iterations=5)
mask_closed_Elipse=cv2.morphologyEx(mask_Opened_Rect,cv2.MORPH_CLOSE,Kernel_Elipse15,iterations=3)
mask_Gradient_Elipse=cv2.morphologyEx(mask_closed_Elipse,cv2.MORPH_GRADIENT,Kernel_Elipse15,iterations=1)


fig,axes=plt.subplots(1,4,figsize=(12,8))
for ax,ch,name in zip(axes,[mask,mask_Opened_Rect,mask_closed_Elipse,mask_Gradient_Elipse],['mask','mask_Opened_Rect','mask_closed_Elipse','mask_Gradient_Elipse']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()

Kernel_Elipse45=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
img_TopHat=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-28 at 19.50.24.jpeg")
img_BlackHat=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-28 at 19.34.45.jpeg")

img_TopHatGray=cv2.cvtColor(img_TopHat,cv2.COLOR_BGR2GRAY)
img_BlackHatGray=cv2.cvtColor(img_BlackHat,cv2.COLOR_BGR2GRAY)

img_TopHatResult=cv2.morphologyEx(img_TopHatGray,cv2.MORPH_TOPHAT,Kernel_Elipse45)
img_BlackHatResult=cv2.morphologyEx(img_BlackHatGray,cv2.MORPH_BLACKHAT,Kernel_Elipse45)





fig,axes=plt.subplots(1,4,figsize=(12,8))
for ax,ch,name in zip(axes,[img_TopHatGray,img_TopHatResult,img_BlackHatGray,img_BlackHatResult],['img_TopHatGray','img_TopHatResult','img_BlackHatGray','img_BlackHatResult']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()

