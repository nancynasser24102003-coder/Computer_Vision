import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
H_color,W_color,Channels_color=img.shape
N_bytes_Color=H_color*W_color*Channels_color
print(f"The Number of Bytes :{N_bytes_Color}")

H_gray,W_gray=img_gray.shape
N_bytes_gray=H_gray*W_gray
print(f"The Number of Bytes :{N_bytes_gray}")


print(f"The Ratio Between The Num Of Bytes of Color Image To Gray Image is : {N_bytes_Color/N_bytes_gray}")

if img.nbytes==N_bytes_Color:
    print("True")
if img_gray.nbytes==N_bytes_gray:
    print("True")

Ratio_nbytes=img.nbytes/img_gray.nbytes
if Ratio_nbytes==N_bytes_Color/N_bytes_gray:
    print("True")








