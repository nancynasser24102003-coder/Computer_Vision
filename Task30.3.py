import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os 

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_SizeinHard=os.path.getsize(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
print(f"The Size of Image in Hard :{img_SizeinHard}")
img_SizeinMemory=img.nbytes
print(f"The Size of Image in Memory : {img_SizeinMemory}")
Ratio_SizeinMemorytoHard=img_SizeinMemory/img_SizeinHard
print(f"The Ratio Between The Size of Image in Memory to in Hard:{Ratio_SizeinHardtoMemory}")

if Ratio_SizeinMemorytoHard>1:
    print("The Size of Image in Memory > The Size of Image in Hard")
elif Ratio_SizeinMemorytoHard==1:
    print("The Size of Image in Memory = The Size of Image in Hard")
else:
    print("The Size of Image in Memory < The Size of Image in Hard")


if img_SizeinMemory > img_SizeinHard:
    print("Memory size is larger than file size because the image is decompressed when loaded into RAM.")



