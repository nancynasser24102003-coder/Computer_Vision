import cv2
import numpy as np
import matplotlib.pyplot as plt 


img_ForGround=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_BackGround=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232318_ChatGPT.jpg")


if img_BackGround is None or img_ForGround is None:
    raise FileNotFoundError("Check Path")

H_ForGround,W_ForGround=img_ForGround.shape[:2]
img_BackGround_Resized=cv2.resize(img_BackGround,(W_ForGround,H_ForGround),cv2.INTER_LINEAR)

img_Combined_Alpha0=cv2.addWeighted(img_ForGround,0,img_BackGround_Resized,1,0)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\fade_0.jpg",img_Combined_Alpha0)
img_Combined_Alpha25=cv2.addWeighted(img_ForGround,0.25,img_BackGround_Resized,0.75,0)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\fade_1.jpg",img_Combined_Alpha25)
img_Combined_Alpha50=cv2.addWeighted(img_ForGround,0.5,img_BackGround_Resized,0.5,0)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\fade_2.jpg",img_Combined_Alpha50)
img_Combined_Alpha75=cv2.addWeighted(img_ForGround,0.75,img_BackGround_Resized,0.25,0)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\fade_3.jpg",img_Combined_Alpha75)
img_Combined_Alpha1=cv2.addWeighted(img_ForGround,1,img_BackGround_Resized,0,0)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\fade_4.jpg",img_Combined_Alpha1)

fig,axes=plt.subplots(1,5,figsize=(14,14))
for ax,ch,name in zip(axes,[img_Combined_Alpha0,img_Combined_Alpha25,img_Combined_Alpha50,img_Combined_Alpha75,img_Combined_Alpha1],['Alpha=0.0','Alpha=0.25','Alpha=0.5','Alpha=0.75','Alpha=1.0']):
    ax.imshow(cv2.cvtColor(ch,cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Combined_Figure.jpg")
plt.show()



















