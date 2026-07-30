import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_all_spaces(path):

  img_BGR=cv2.imread(path)
  if img_BGR is None:
    raise FileNotFoundError("The image not loaded")
  
  img_GRAY=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2GRAY)
  img_HSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
  img_LAB=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2LAB)
  img_YCrCb=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2YCrCb)

  H,W= img_BGR.shape[:2]
  Cx,Cy=W//2, H//2


  print(f"The Pixels Center Values of BGR Image: {img_BGR[Cy,Cx]}")


  print(f"The Pixels Center Values of GRAY Image: {img_GRAY[Cy,Cx]}")

 
  print(f"The Pixels Center Values of HSV Image: {img_HSV[Cy,Cx]}")


  print(f"The Pixels Center Values of LAB Image: {img_LAB[Cy,Cx]}")

  print(f"The Pixels Center Values of YCrCb Image: {img_YCrCb[Cy,Cx]}")



  plt.figure(figsize=(14,14))
  plt.subplot(1,5,1)
  plt.imshow(cv2.cvtColor(img_BGR,cv2.COLOR_BGR2RGB))
  plt.title("BGR (Original)")
  plt.axis("off")
  plt.subplot(1,5,2)
  plt.imshow(img_GRAY,cmap='gray')
  plt.title("Gray image")
  plt.axis("off")
  plt.subplot(1,5,3)
  plt.imshow(img_HSV)
  plt.title("HSV image")
  plt.axis("off")
  plt.subplot(1,5,4)
  plt.imshow(img_LAB)
  plt.title("LAB image")
  plt.axis("off")
  plt.subplot(1,5,5)
  plt.imshow(img_YCrCb)
  plt.title("YCrCb")
  plt.axis("off")
  plt.tight_layout()
  plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Color_Spaces.png")
  plt.show()


show_all_spaces(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")





















