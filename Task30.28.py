import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 01.32.39.jpeg")
if img is None :
    raise FileNotFoundError("The image not Loaded ")

Colors=['b','g','r']
Labels=['Blue','Green','Red']
Peak_count=np.array([0.0,0.0,0.0])
Peak_Value=np.array([0,0,0])

for i,(col,lab) in enumerate(zip(Colors,Labels)):
    hist=cv2.calcHist([img],[i],None,[256],[0,256])
    Peak_count[i]=hist.max()
    Peak_Value[i]=hist.argmax()
    plt.plot(hist.flatten(),color=col,label=lab)
plt.legend()
plt.xlabel("Intensity")
plt.ylabel("Num of Pixels")
plt.xlim([0,256])
plt.show()
print(f"The Peak_Count in ['b','g','r']: {Peak_count}")
print(f"The Peak_Value in ['b','g','r']: {Peak_Value}")

Index_Max=np.argmax(Peak_count)

print(f"Max Peak Count: {Peak_count[Index_Max]}")
print(f"Max Peak Value: {Peak_Value[Index_Max]}")
print(f"Max Channel: {Labels[Index_Max]}")













