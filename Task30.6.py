import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
roi=img[:,:].copy()

roi[0:40,0:40]=[0,0,255]

H_roi,W_roi=roi.shape[:2]

roi[H_roi//2,:]=[255,255,255]


roi[:,W_roi//2]=[255,255,255]

roi[100:200,900:1000]=[0,255,0]

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("img_Orig")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
plt.title("img_Edit")
plt.axis("off")
plt.tight_layout()
plt.show()
