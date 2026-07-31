import cv2
import numpy as np
import matplotlib.pyplot as plt 


img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 20.52.46.jpeg")

if img is None :
    raise FileNotFoundError("The Image not Loaded")

print(f"The Shape of Image {img.shape}")

Reference_Patch=img[360:400,360:400].copy()


img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

Reference_Patch_HSV=cv2.cvtColor(Reference_Patch,cv2.COLOR_BGR2HSV)

Hist_ReferencePatch=cv2.calcHist([Reference_Patch_HSV],[0,1],None,[180,256],[0,180,0,256])

ReferencePatch_Normalized=cv2.normalize(Hist_ReferencePatch,Hist_ReferencePatch,0,256,cv2.NORM_MINMAX)

Probability_Map=cv2.calcBackProject([img_HSV],[0,1],ReferencePatch_Normalized,[0,180,0,256],1)

Kernel_Elipse5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

Probability_Map=cv2.filter2D(Probability_Map,-1,Kernel_Elipse5)

T,mask=cv2.threshold(Probability_Map,100,255,cv2.THRESH_BINARY)

Result=cv2.bitwise_and(img,img,mask=mask)


plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(Reference_Patch, cv2.COLOR_BGR2RGB))
plt.title("Reference Patch")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Full Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(Probability_Map, cmap="gray")
plt.title("Probability Map")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(Result, cv2.COLOR_BGR2RGB))
plt.title("Final Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
















































