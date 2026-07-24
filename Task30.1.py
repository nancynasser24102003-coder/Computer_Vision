import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
print(f"The Shape of Color Image :{img.shape}\nThe Data Type of Color Image :{img.dtype}\nThe Number of Array of Color Image :{img.ndim}\nThe Size of Color Image :{img.size}")

img_GRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
H_GRAY,W_GRAY=img_GRAY.shape
H,W=img.shape[:2]
print(f"The value of Center Pixel of Color image : {img[H//2,W//2]} ")
print(f"The value of Center Pixel of gray image : {img_GRAY[H_GRAY//2,W_GRAY//2]} ")
roi=img[350:800,300:800]
roi_GRAY=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
roi_GRAY_3ch=cv2.cvtColor(roi_GRAY,cv2.COLOR_GRAY2BGR)
roi[:,:]=roi_GRAY_3ch
cv2.imshow("ROI",roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("ROI_Resulted",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
roi=img[350:800,300:800]
roi[:,:]=[255,255,255]
cv2.imshow("ROI",roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("ROI_Resulted",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_GRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Orig_img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Gray_img",img_GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"SCALAR   -> img_GRAY[100,200]= {img_GRAY[100,200]}           shape :{img_GRAY[100,200].shape}")
print(f"VECTOR   -> img[100,200]= {img[100,200]}                     shape :{img[100,200].shape}")
print(f"MATRIX   -> img_GRAY.shape= {img_GRAY.shape}                 shape :{img_GRAY.shape}")
print(f"3D ARRAY -> img.shape= {img.shape}                           shape :{img.shape}")







