import cv2
import numpy as np
import matplotlib.pyplot as plt 


Lower_Range1=np.array([0,70,50])
Upper_Range1=np.array([10,255,255])

Lower_Range2=np.array([160,70,50])
Upper_Range2=np.array([179,255,255])

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-30 at 11.45.27.jpeg")
if img is None:
    raise FileNotFoundError("The image not loaded")

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask1=cv2.inRange(img_hsv,Lower_Range1,Upper_Range1)
mask2=cv2.inRange(img_hsv,Lower_Range2,Upper_Range2)

CorrectMask=cv2.bitwise_or(mask1,mask2)
NaiveMask=mask1

img_red_correct=cv2.bitwise_and(img,img,mask=CorrectMask)
img_red_naive=cv2.bitwise_and(img,img,mask=NaiveMask)





fig,axes=plt.subplots(2,3,figsize=(8,8))
axes[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0,0].set_title("Original Image")
axes[0,0].axis("off")
axes[0,1].imshow(CorrectMask,cmap='gray')
axes[0,1].set_title("Correct Mask")
axes[0,1].axis("off")
axes[0,2].imshow(cv2.cvtColor(img_red_correct,cv2.COLOR_BGR2RGB))
axes[0,2].set_title("Correct Segmented Result")
axes[0,2].axis("off")
axes[1,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[1,0].set_title("Original Image")
axes[1,0].axis("off")
axes[1,1].imshow(NaiveMask,cmap='gray')
axes[1,1].set_title("Naive Mask")
axes[1,1].axis("off")
axes[1,2].imshow(cv2.cvtColor(img_red_naive,cv2.COLOR_BGR2RGB))
axes[1,2].set_title("Correct Segmented Result")
axes[1,2].axis("off")
plt.tight_layout()
plt.show()
























