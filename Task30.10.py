import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
if img is None :
    raise FileExistsError("The Img not Loaded")

H,W=img.shape[:2]
Cx,Cy=W//2,H//2
r=min(H,W)//8

y,x=np.ogrid[:H,:W]
mask=(x-Cx)**2+(y-Cy)**2<=r**2
img[mask]=[255,255,255]



r_inner=r+20
r_outer=r+40
mask_outer=(x-Cx)**2+(y-Cy)**2<=r_outer**2
mask_inner=(x-Cx)**2+(y-Cy)**2<=r_inner**2
mask_ring = mask_outer & ~mask_inner
img[mask_ring]=[0,0,255]
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\img.png",img)
cv2.imshow("img_mask",img)
cv2.waitKey(0)
cv2.destroyAllWindows()










