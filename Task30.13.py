import cv2
import numpy as np
import matplotlib.pyplot as plt 

Lower_Range=np.array([0,133,77])
Upper_Range=np.array([255,173,127])

img=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
if img is None :
    raise FileNotFoundError("The Image Not Loaded ")
img_YCrCb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

mask_Skin=cv2.inRange(img_YCrCb,Lower_Range,Upper_Range)

Skin_only_Result=cv2.bitwise_and(img,img,mask=mask_Skin)

plt.figure(figsize=(14,14))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(mask_Skin,cmap='gray')
plt.title("Skin Mask")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(Skin_only_Result,cv2.COLOR_BGR2RGB))
plt.title("Skin_only_Result")
plt.axis("off")
plt.tight_layout()
plt.show()




