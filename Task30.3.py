import cv2
import numpy as np 
import matplotlib.pyplot as plt



arr_uint8=np.array([250,245,4],dtype=np.uint8)
arr_uint8=arr_uint8+15
print(arr_uint8)



arr_uint8=np.array([250,245,4],dtype=np.uint8)
arr_float32=arr_uint8.astype(np.float32)
arr_float32=arr_float32+15.0
print(arr_float32)


if arr_float32.max() > 255:
    print("float32 stores values greater than 255 without overflow.")
if arr_uint8.max() <255:
    print("uint8 wraps around when values exceed 255, causing overflow.")










