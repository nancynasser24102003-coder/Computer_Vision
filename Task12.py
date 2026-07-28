import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-28 at 17.12.50.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
T,mask=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)

Kernel_Rect3=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
Kernel_Elipse5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


img_grayerode_rect3=cv2.erode(img_gray,Kernel_Rect3,iterations=7)
img_grayerode_elipse5=cv2.erode(img_gray,Kernel_Elipse5,iterations=5)
img_graydilate_rect3=cv2.dilate(img_gray,Kernel_Rect3,iterations=5)
img_graydilate_elipse5=cv2.dilate(img_gray,Kernel_Elipse5,iterations=5)
img_grayopen_rect3=cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,Kernel_Rect3,iterations=7)
img_grayopen_elipse5=cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,Kernel_Elipse5,iterations=5)
img_grayclose_rect3=cv2.morphologyEx(img_gray,cv2.MORPH_CLOSE,Kernel_Rect3,iterations=7)
img_grayclose_elipse5=cv2.morphologyEx(img_gray,cv2.MORPH_CLOSE,Kernel_Elipse5,iterations=5)
img_graygradient_rect3=cv2.morphologyEx(img_gray,cv2.MORPH_GRADIENT,Kernel_Rect3,iterations=5)
img_graytophat=cv2.morphologyEx(img_gray,cv2.MORPH_TOPHAT,Kernel_Elipse5,iterations=5)
img_grayblackhat=cv2.morphologyEx(img_gray,cv2.MORPH_BLACKHAT,Kernel_Elipse5,iterations=5)


fig,axes=plt.subplots(2,6,figsize=(14,8))
axes[0,0].imshow(img_gray,cmap='gray')
axes[0,0].set_title("img_gray")
axes[0,0].axis("off")
axes[0,1].imshow(img_grayerode_rect3,cmap='gray')
axes[0,1].set_title("img_grayerode_rect3")
axes[0,1].axis("off")
axes[0,2].imshow(img_grayerode_elipse5,cmap='gray')
axes[0,2].set_title("img_grayerode_elipse5")
axes[0,2].axis("off")
axes[0,3].imshow(img_graydilate_rect3,cmap='gray')
axes[0,3].set_title("img_graydilate_rect3")
axes[0,3].axis("off")
axes[0,4].imshow(img_graydilate_elipse5,cmap='gray')
axes[0,4].set_title("img_graydilate_elipse5")
axes[0,4].axis("off")
axes[0,5].imshow(img_grayopen_rect3,cmap='gray')
axes[0,5].set_title("img_grayopen_rect3")
axes[0,5].axis("off")
axes[1,0].imshow(img_grayopen_elipse5,cmap='gray')
axes[1,0].set_title("img_grayopen_elipse5")
axes[1,0].axis("off")
axes[1,1].imshow(img_grayclose_rect3,cmap='gray')
axes[1,1].set_title("img_grayclose_rect3")
axes[1,1].axis("off")
axes[1,2].imshow(img_grayclose_elipse5,cmap='gray')
axes[1,2].set_title("img_grayclose_elipse5")
axes[1,2].axis("off")
axes[1,3].imshow(img_graygradient_rect3,cmap='gray')
axes[1,3].set_title("img_graygradient_rect3")
axes[1,3].axis("off")
axes[1,4].imshow(img_grayblackhat,cmap='gray')
axes[1,4].set_title("img_grayblackhat")
axes[1,4].axis("off")
axes[1,5].imshow(img_graytophat,cmap='gray')
axes[1,5].set_title("img_graytophat")
axes[1,5].axis("off")
plt.tight_layout()
plt.show()


img_MaskOpen_elipse5=cv2.morphologyEx(mask,cv2.MORPH_OPEN,Kernel_Elipse5,iterations=5)
img_MaskClose_elipse5=cv2.morphologyEx(img_MaskOpen_elipse5,cv2.MORPH_CLOSE,Kernel_Elipse5,iterations=5)
mask_clean=img_MaskClose_elipse5

fig,axes=plt.subplots(1,3,figsize=(12,8))
for ax,ch,name in zip(axes,[mask,img_MaskOpen_elipse5,img_MaskClose_elipse5],['mask','img_MaskOpen_elipse5','img_MaskClose_elipse5']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\mask_cleanup.png")
plt.show()

def validate(mask):
    print(f"The Shape of Clean Mask: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    print(f"Unique Values: {np.unique(mask)}")
    


validate(mask_clean)






