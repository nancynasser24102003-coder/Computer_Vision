import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-21 at 16.44.45.jpeg")
Spaces={
    'RGB':cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
    'HSV':cv2.cvtColor(img,cv2.COLOR_BGR2HSV),
    'LAB':cv2.cvtColor(img,cv2.COLOR_BGR2LAB),
    'Ycrcb':cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb),
    'Gray':cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
    'BGR':img,
    
}

fig,axes=plt.subplots(1,5,figsize=(8,8))
for ax,ch,name in zip(axes,Spaces.values(),Spaces.keys()):

    if name=='Gray':
       ax.imshow(ch,cmap='gray')

    else:
      ax.imshow(ch)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()


for name,ch in zip(Spaces.keys(),Spaces.values()):
   H,W=ch.shape[:2]
   Cx,Cy=W//2,H//2

   print(f"Center Pixels of {name}:{ch[Cy,Cx]}")


img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_range=np.array([0,100,100])
upper_range=np.array([10,255,255])
mask=cv2.inRange(img_HSV,lower_range,upper_range)
result=cv2.bitwise_and(img,img,mask=mask)
img_RGB_edit=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
img_original=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.subplot(1,3,1)
plt.imshow(mask,cmap='gray')
plt.title("mask")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(img_original)
plt.title("original")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(img_RGB_edit)
plt.title("segmented result")
plt.axis("off")
plt.tight_layout()
plt.show()