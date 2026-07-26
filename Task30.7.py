import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
if img is None :
    raise FileNotFoundError("The Image not Loaded")

B,G,R=cv2.split(img)
fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax ,ch , name in zip(axes,[B,G,R],['Blue','Green','Red']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\task7.1.png")
plt.show()
H,W=img.shape[:2]
arr_zeros=np.zeros((H,W),dtype=np.uint8)
Blue=cv2.merge([B,arr_zeros,arr_zeros])
Green=cv2.merge([arr_zeros,G,arr_zeros])
Red=cv2.merge([arr_zeros,arr_zeros,R])

fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax ,ch , name in zip(axes,[Blue,Green,Red],['Blue_only','Green_only','Red_only']):
    ax.imshow(cv2.cvtColor(ch,cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\task7.2.png")
plt.show()












