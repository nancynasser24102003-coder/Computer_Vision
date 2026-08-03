import cv2
import numpy as np
import matplotlib.pyplot as plt 

Lower_Red1=np.array([0,70,50])
Upper_Red1=np.array([10,255,255])
Lower_Red2=np.array([170,70,50])
Upper_Red2=np.array([179,255,255])

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 20.52.46.jpeg")

if img is None :
    raise FileNotFoundError("The Image not Loaded")


img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
Mask_Red1=cv2.inRange(img_HSV,Lower_Red1,Upper_Red1)
Mask_Red2=cv2.inRange(img_HSV,Lower_Red2,Upper_Red2)
Mask_Result=cv2.bitwise_or(Mask_Red1,Mask_Red2)
Blue_Background=np.full_like(img,(255,0,0))
# Blue_Background=np.full((h,w,3),(255,0,0),dtype=np.uint8)
Apple=cv2.bitwise_and(img,img,mask=Mask_Result)
Mask_Inv=cv2.bitwise_not(Mask_Result)
Blue_Background_Result=cv2.bitwise_and(Blue_Background,Blue_Background,mask=Mask_Inv)
Apple_Result=cv2.add(Blue_Background_Result,Apple)




plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(Mask_Result,cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(Apple_Result,cv2.COLOR_BGR2RGB))
plt.title("Composite")
plt.axis("off")

plt.tight_layout()
plt.show()



















