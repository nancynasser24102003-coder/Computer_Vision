import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-27 at 14.08.27.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
T=70
T,img_Binary=cv2.threshold(img_gray,T,255,cv2.THRESH_BINARY)
T,img_BinaryInv=cv2.threshold(img_gray,T,255,cv2.THRESH_BINARY_INV)
T,img_Trunc=cv2.threshold(img_gray,T,255,cv2.THRESH_TRUNC)
T,img_ToZero=cv2.threshold(img_gray,T,255,cv2.THRESH_TOZERO)
T,img_ToZeroInv=cv2.threshold(img_gray,T,255,cv2.THRESH_TOZERO_INV)


fig,axes=plt.subplots(2,3,figsize=(8,8))
axes[0,0].imshow(img_gray,cmap='gray')
axes[0,0].set_title("img_gray")
axes[0,0].axis("off")
axes[0,1].imshow(img_Binary,cmap='gray')
axes[0,1].set_title("img_Binary")
axes[0,1].axis("off")
axes[0,2].imshow(img_BinaryInv,cmap='gray')
axes[0,2].set_title("img_BinaryInv")
axes[0,2].axis("off")
axes[1,0].imshow(img_Trunc,cmap='gray')
axes[1,0].set_title("img_Trunc")
axes[1,0].axis("off")
axes[1,1].imshow(img_ToZero,cmap='gray')
axes[1,1].set_title("img_ToZero")
axes[1,1].axis("off")
axes[1,2].imshow(img_ToZeroInv,cmap='gray')
axes[1,2].set_title("img_ToZeroInv")
axes[1,2].axis("off")
plt.tight_layout()
plt.show()


T_Otsu,img_BinaryOtsu=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(f"T_Otsu : {T_Otsu}")
cv2.imshow("img_BinaryOtsu",img_BinaryOtsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(hist.flatten(),color='black')
plt.axvline(x=T_Otsu,color='red',linestyle='--',label="Otsu Threshold")
plt.legend()
plt.title("img_GrayHist")
plt.xlim([0,256])
plt.show()

img_AdaptiveMean=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=11,C=2)
img_AdaptiveGaussian=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=11,C=2)
fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax,ch,name in zip(axes,[img_gray,img_AdaptiveMean,img_AdaptiveGaussian],['img_gray','img_AdaptiveMean','img_AdaptiveGaussian']):
    ax.imshow(ch,cmap='gray')
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()

Lower_Range=np.array([40,50,100])
Upper_Range=np.array([75,255,255])
cap=cv2.VideoCapture(0)
while True :
    ret,image=cap.read()
    if not ret:
        break

    blurred=cv2.GaussianBlur(image,(3,3),0)
    img_hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(img_hsv,Lower_Range,Upper_Range)

    result=cv2.bitwise_and(image,image,mask=mask)
    Combined=np.hstack([image,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),result])
    Combined=cv2.resize(Combined,(1200,400))
    cv2.imshow('|   img_Orig   |,|   Mask   |,|   Segmented   |',Combined)

    if cv2.waitKey(1) & 0xFF==ord('c'):
        break


cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\image.png",image)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\mask.png",mask)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\result.png",result)

cap.release()
cv2.destroyAllWindows()


















