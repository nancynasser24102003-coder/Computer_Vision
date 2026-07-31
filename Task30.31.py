import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 14.29.00.jpeg")
if img is None :
    raise FileNotFoundError("The Image not Loaded ")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Clip_Limit_List=[1.0,2.0,8.0,40.0]
img_Clahed=[]

for i in Clip_Limit_List:
    Clahe=cv2.createCLAHE(clipLimit=i,tileGridSize=(8,8))
    Result=Clahe.apply(img_gray)
    img_Clahed.append(Result)

fig,axes=plt.subplots(1,4,figsize=(16,4))
for ax,ch,name in zip(axes,img_Clahed,['ClipLimit=1.0','ClipLimit=2.0','ClipLimit=8.0','ClipLimit=40.0']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.suptitle("CLAHE with Different Clip Limits")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Different_Clip_Limit.png")
plt.show()

