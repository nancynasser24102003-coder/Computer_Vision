import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img_Dark=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 00.29.08.jpeg")
img_Bright=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 00.29.07.jpeg")
img_Normal=cv2.imread(r"C:\Users\Etijah\Pictures\WhatsApp Image 2026-07-31 at 00.29.07.jpeg")

img_Dark_togray=cv2.cvtColor(img_Dark,cv2.COLOR_BGR2GRAY)
img_Bright_togray=cv2.cvtColor(img_Bright,cv2.COLOR_BGR2GRAY)
img_Normal_togray=cv2.cvtColor(img_Normal,cv2.COLOR_BGR2GRAY)

def diagnose(img_gray):
    img_mean=img_gray.mean()
    img_std=img_gray.std()
    if img_mean < 80:
        return "The Gray Image is Underexposed "
    elif img_mean > 180:
        return "The Gray Image is Overexposed "
    elif img_std < 30:
        return "The Gray Image is Low Contrast "
    else :
        return "The Gray Image is OK "




hist_Dark=cv2.calcHist([img_Dark_togray],[0],None,[256],[0,256])
hist_Bright=cv2.calcHist([img_Bright_togray],[0],None,[256],[0,256])
hist_Normal=cv2.calcHist([img_Normal_togray],[0],None,[256],[0,256])


fig,axes=plt.subplots(2,3,figsize=(8,8))
axes[0,0].imshow(img_Dark_togray,cmap='gray')
axes[0,0].set_title(diagnose(img_Dark_togray))
axes[0,0].axis("off")
axes[0,1].imshow(img_Bright_togray,cmap='gray')
axes[0,1].set_title(diagnose(img_Bright_togray))
axes[0,1].axis("off")
axes[0,2].imshow(img_Normal_togray,cmap='gray')
axes[0,2].set_title(diagnose(img_Normal_togray))
axes[0,2].axis("off")
axes[1,0].plot(hist_Dark.flatten(),color='red')
axes[1,0].set_xlim([0,256])
axes[1,0].set_title("Dark Histogram")
axes[1,1].plot(hist_Bright.flatten(),color='blue')
axes[1,1].set_xlim([0,256])
axes[1,1].set_title("Bright Histogram")
axes[1,2].plot(hist_Normal.flatten(),color='green')
axes[1,2].set_xlim([0,256])
axes[1,2].set_title("Normal Histogram")
plt.suptitle("Histogram Shape Diagnosis")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Histogram_Diagnosis.png")
plt.show()















