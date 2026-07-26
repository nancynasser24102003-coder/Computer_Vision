import cv2
import numpy as np
import matplotlib.pyplot as plt 
img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
if img is None :
    raise FileNotFoundError("The Image not Loaded")


img_Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_AverageGray=img.mean(axis=2)
img_AverageGray_uint8=img_AverageGray.astype(np.uint8)
img_Differnce=cv2.absdiff(img_Gray,img_AverageGray_uint8)

fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax ,ch , name in zip(axes,[img_Gray,img_AverageGray_uint8,img_Differnce],['img_Gray','img_AverageGray_uint8','img_Differnce']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\task8.png")
plt.show()

img_Differnce_Max=img_Differnce.max()
img_Differnce_mean=img_Differnce.mean()

print(f"The Max Value in img_Differnce :{img_Differnce_Max} \nThe Mean Value of img_Differnce: {img_Differnce_mean} ")












