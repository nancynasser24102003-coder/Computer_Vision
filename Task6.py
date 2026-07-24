import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg") 
roi=img[350:800,300:800].copy()
img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
roi_RGB=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
print(f"The of size img : {img.shape}")
plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plt.imshow(img_RGB)
plt.axis("off")
plt.title("Orig_img")
plt.subplot(1,2,2)
plt.imshow(roi_RGB)
plt.axis("off")
plt.title("Crop_img")
plt.tight_layout()
plt.show()


H_roi,W_roi=roi.shape[:2]
cv2.resize(roi_RGB,(W_roi,H_roi),cv2.INTER_LINEAR)
plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.imshow(cv2.resize(roi_RGB,(200,200),interpolation=cv2.INTER_LINEAR))
plt.axis("off")
plt.title("roi_200*200")
plt.subplot(1,3,2)
plt.imshow(cv2.resize(roi_RGB,(W_roi//2,H_roi//2),interpolation=cv2.INTER_AREA))
plt.axis("off")
plt.title("roi_HalfScale")
plt.subplot(1,3,3)
plt.imshow(cv2.resize(roi_RGB,(W_roi*2,H_roi*2),interpolation=cv2.INTER_CUBIC))
plt.axis("off")
plt.title("roi_DoubleScale")
plt.tight_layout()
plt.show()

# cv2.imshow("roi_200*200",cv2.resize(roi,(200,200),interpolation=cv2.INTER_LINEAR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("roi_HalfScale",cv2.resize(roi,(W_roi//2,H_roi//2),interpolation=cv2.INTER_LINEAR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("roi_DoubleScale",cv2.resize(roi,(W_roi*2,H_roi*2),interpolation=cv2.INTER_LINEAR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
H,W=img.shape[:2]
M_30=cv2.getRotationMatrix2D((W//2,H//2),30,1.0)
img_rotated30=cv2.warpAffine(img,M_30,(W,H))
M_Minus30=cv2.getRotationMatrix2D((W//2,H//2),-30,1.0)
img_rotatedminus30=cv2.warpAffine(img,M_Minus30,(W,H))
plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_orig")
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img_rotated30,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_rotated30")
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(img_rotatedminus30,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_rotated-30")
plt.tight_layout()
plt.show()





img_Flip0=cv2.flip(img,0)
img_Flip1=cv2.flip(img,1)
img_Flipminus1=cv2.flip(img,-1)
plt.figure(figsize=(12,12))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_orig")
plt.subplot(1,4,2)
plt.imshow(cv2.cvtColor(img_Flip0,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_rotated30")
plt.subplot(1,4,3)
plt.imshow(cv2.cvtColor(img_Flip1,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_rotated-30")
plt.subplot(1,4,4)
plt.imshow(cv2.cvtColor(img_Flipminus1,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_rotated-30")
plt.tight_layout()
plt.show()



img_background=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232318_ChatGPT.jpg")
H,W=img.shape[:2]
img_Resized=cv2.resize(img_background,(W,H),interpolation=cv2.INTER_LINEAR)
img_Blended=cv2.addWeighted(img,0.8,img_Resized,0.2,0)
cv2.imshow("img_Blended",img_Blended)
cv2.waitKey(0)
cv2.destroyAllWindows()


H,W=img.shape[:2]
img_black=np.zeros((H,W),dtype=np.uint8)
mask=cv2.circle(img_black,(500,500),300,255,-1)
mask_resulted=cv2.bitwise_and(img,img,mask=mask)
plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("img_orig")
plt.subplot(1,3,2)
plt.imshow(mask,cmap='gray')
plt.axis("off")
plt.title("mask")
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(mask_resulted,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("mask_resulted")
plt.tight_layout()
plt.show()









