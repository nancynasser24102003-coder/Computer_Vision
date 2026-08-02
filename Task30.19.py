import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img_bgr=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-08-02 at 14.10.29.jpeg")
if img_bgr is None :
    raise FileNotFoundError("The Image not Loaded")

img_lab=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2LAB)
L,A,B_LAB=cv2.split(img_lab)
B,G,R=cv2.split(img_bgr)
Clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
L_enh=Clahe.apply(L)
B_enh=Clahe.apply(B)
G_enh=Clahe.apply(G)
R_enh=Clahe.apply(R)
img_bgr_enh=cv2.merge([B_enh,G_enh,R_enh])
img_lab_enh=cv2.merge([L_enh,A,B_LAB])
images=[cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB),cv2.cvtColor(img_bgr_enh,cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.cvtColor(img_lab_enh,cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)]
image_names=["Original Image","Enhanced Image with BGR","Enhanced Image with LAB"]
fig,axes=plt.subplots(1,3,figsize=(12,12))
for ax ,ch,name in zip(axes,images,image_names):
    ax.imshow(ch)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()







