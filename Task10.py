import cv2
import numpy as np
import matplotlib.pyplot as plt


Kernel_Rect5=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
Kernel_Elipse5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
Kernel_Cross5=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
Kernel_CustomDiamoud=np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=np.uint8)
print("Rectangle Kernel")
print(Kernel_Rect5)
print("Elipse Kernel")
print(Kernel_Elipse5)
print("Cross Kernel")
print(Kernel_Cross5)
print("Diamound Kernel")
print(Kernel_CustomDiamoud)

















