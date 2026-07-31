import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 14.29.00.jpeg")
if img is None :
    raise FileNotFoundError("The Image not Loade ")



img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_Equalized=cv2.equalizeHist(img_gray)
Clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img_Clahed=Clahe.apply(img_gray)




hist_gray=cv2.calcHist([img_gray],[0],None,[256],[0,256])
hist_Equalized=cv2.calcHist([img_Equalized],[0],None,[256],[0,256])
hist_Clahed=cv2.calcHist([img_Clahed],[0],None,[256],[0,256])





plt.figure(figsize=(12,12))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Original Image ")
plt.subplot(1,4,2)
plt.imshow(img_gray,cmap='gray')
plt.axis("off")
plt.title(" Gray Image ")
plt.subplot(1,4,3)
plt.imshow(img_Equalized,cmap='gray')
plt.axis("off")
plt.title(" Equalized Image  ")
plt.subplot(1,4,4)
plt.imshow(img_Clahed,cmap='gray')
plt.axis("off")
plt.title(" Clahed Image  ")
plt.tight_layout()
plt.show()




plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.plot(hist_gray.flatten(),color='red')
plt.xlim([0,256])
plt.title("Histogram of Gray Image ")
plt.subplot(1,3,2)
plt.plot(hist_Equalized.flatten(),color='blue')
plt.xlim([0,256])
plt.title(" Histogram of Equalized Image ")
plt.subplot(1,3,3)
plt.plot(hist_Clahed.flatten(),color='black')
plt.xlim([0,256])
plt.title(" Histogram of Clahed Image  ")
plt.tight_layout()
plt.show()




























































































