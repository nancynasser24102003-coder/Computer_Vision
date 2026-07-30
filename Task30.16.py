import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
H,W=img.shape[:2]
Cx,Cy=W//2,H//2
M_30=cv2.getRotationMatrix2D((Cx,Cy),30,1)
img_Rotated30=cv2.warpAffine(img,M_30,(W,H),borderMode=cv2.BORDER_REPLICATE)
M_Minus30=cv2.getRotationMatrix2D((Cx,Cy),-30,1)
img_RotatedMinus30=cv2.warpAffine(img,M_Minus30,(W,H),borderMode=cv2.BORDER_REPLICATE)
img_Flip0=cv2.flip(img,0)
img_Flip1=cv2.flip(img,1)
img_FlipMinus1=cv2.flip(img,-1)



fig,axes=plt.subplots(2,3,figsize=(12,8))
axes[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0,0].set_title("Original Image")
axes[0,0].axis("off")
axes[0,1].imshow(cv2.cvtColor(img_Rotated30,cv2.COLOR_BGR2RGB))
axes[0,1].set_title("Rotated +30")
axes[0,1].axis("off")
axes[0,2].imshow(cv2.cvtColor(img_RotatedMinus30,cv2.COLOR_BGR2RGB))
axes[0,2].set_title("Rotated -30")
axes[0,2].axis("off")
axes[1,0].imshow(cv2.cvtColor(img_Flip0,cv2.COLOR_BGR2RGB))
axes[1,0].set_title("Vertical Flip")
axes[1,0].axis("off")
axes[1,1].imshow(cv2.cvtColor(img_Flip1,cv2.COLOR_BGR2RGB))
axes[1,1].set_title("Horizontal Flip")
axes[1,1].axis("off")
axes[1,2].imshow(cv2.cvtColor(img_FlipMinus1,cv2.COLOR_BGR2RGB))
axes[1,2].set_title("Both Axes Flip")
axes[1,2].axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\RotatedImg.jpg")
plt.show()







