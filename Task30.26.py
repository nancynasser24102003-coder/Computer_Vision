import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
if img is None :
    raise FileNotFoundError("The Image not Loaded")


img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(hist.flatten(),color='red')
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Number of Pixels ")
plt.xlim([0,256])
plt.show()

hist_Normalize=cv2.normalize(hist,None)
plt.plot(hist_Normalize.flatten(),color='blue')
plt.title("Normalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Number of Pixels ")
plt.xlim([0,256])
plt.show()


print(f"The Shape Of Histogram : {hist.shape}")
print(f"The ndim {hist.ndim}")



















