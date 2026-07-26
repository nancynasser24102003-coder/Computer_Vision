import cv2
import numpy as np
import matplotlib.pyplot as plt 



def safe_load(path, flag=cv2.IMREAD_COLOR):
    img=cv2.imread(path,flag)
    if img is None :
        raise FileNotFoundError("Failed to load image : the file was not found at this path ")

    
    print(f"The Shape of Image :{img.shape}")
    print(f"The Data Type of Image : {img.dtype}")
    print(f"The Flag of Image is :{flag}")

    

safe_load(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg",cv2.IMREAD_COLOR)
safe_load(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg",cv2.IMREAD_GRAYSCALE)
safe_load(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg",cv2.IMREAD_UNCHANGED)


try:
    safe_load(r"C:\Users\Etijah\Desktop\18576944251041430_2870406156677966472_n-1.jpg",cv2.IMREAD_COLOR)

except FileNotFoundError as error:
    print(error)












