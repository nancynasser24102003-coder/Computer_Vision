import cv2 
import numpy as np
import matplotlib.pyplot as plt 

Lower_OrangeRange=np.array([10,50,50])
Upper_OrangeRange=np.array([25,255,255])

Lower_YellowRange=np.array([25,50,50])
Upper_YellowRange=np.array([35,255,255])

Lower_GreenRange=np.array([35,50,50])
Upper_GreenRange=np.array([85,255,255])

Lower_BlueRange=np.array([85,50,50])
Upper_BlueRange=np.array([130,255,255])


img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-30 at 11.06.23.jpeg")
if img is None:
    raise FileNotFoundError("The image not loaded  ")

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask_Orange=cv2.inRange(img_hsv,Lower_OrangeRange,Upper_OrangeRange)

mask_Yellow=cv2.inRange(img_hsv,Lower_YellowRange,Upper_YellowRange)

mask_Green=cv2.inRange(img_hsv,Lower_GreenRange,Upper_GreenRange)

mask_Blue=cv2.inRange(img_hsv,Lower_BlueRange,Upper_BlueRange)

img_Orange=cv2.bitwise_and(img,img,mask=mask_Orange)

img_Yellow=cv2.bitwise_and(img,img,mask=mask_Yellow)

img_Green=cv2.bitwise_and(img,img,mask=mask_Green)

img_Blue=cv2.bitwise_and(img,img,mask=mask_Blue)

cv2.imshow("Original_img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()




fig,axes=plt.subplots(2,4,figsize=(14,8))
axes[0,0].imshow(mask_Orange,cmap='gray')
axes[0,0].set_title("mask_Orange")
axes[0,0].axis("off")
axes[0,1].imshow(mask_Yellow,cmap='gray')
axes[0,1].set_title("mask_Yellow")
axes[0,1].axis("off")
axes[0,2].imshow(mask_Green,cmap='gray')
axes[0,2].set_title("mask_Green")
axes[0,2].axis("off")
axes[0,3].imshow(mask_Blue,cmap='gray')
axes[0,3].set_title("mask_Blue")
axes[0,3].axis("off")
axes[1,0].imshow(cv2.cvtColor(img_Orange,cv2.COLOR_BGR2RGB))
axes[1,0].set_title("img_Orange")
axes[1,0].axis("off")
axes[1,1].imshow(cv2.cvtColor(img_Yellow,cv2.COLOR_BGR2RGB))
axes[1,1].set_title("img_Yellow")
axes[1,1].axis("off")
axes[1,2].imshow(cv2.cvtColor(img_Green,cv2.COLOR_BGR2RGB))
axes[1,2].set_title("img_Green")
axes[1,2].axis("off")
axes[1,3].imshow(cv2.cvtColor(img_Blue,cv2.COLOR_BGR2RGB))
axes[1,3].set_title("img_Blue")
axes[1,3].axis("off")
plt.tight_layout()
plt.show()



































