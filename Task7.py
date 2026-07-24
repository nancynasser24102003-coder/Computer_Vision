import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_GRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([img_GRAY],[0],None,[256],[0,255])
img_eq=cv2.equalizeHist(img_GRAY)
hist_eq=cv2.calcHist([img_eq],[0],None,[256],[0,255])
plt.plot(hist.flatten(),color='black',linewidth=1.5)
plt.xlim([0,256])
plt.show()
print(f"The Mean of Image:{np.mean(img_GRAY)},The contast og Image: {np.std(img_GRAY)}")


Colors=('b','g','r')
Labels=("Blue","Green","Red")
for i ,(col,lab) in enumerate(zip(Colors,Labels)):
    hist=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist.flatten(),color=col,label=lab)
plt.legend()
plt.xlim([0,256])
plt.show()


H,W=img.shape[:2]
img_Black=np.zeros((H,W),dtype=np.uint8)
mask= cv2.circle(img_Black,(500,500),300,255,-1)
mask_resulted=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("mask_resulted",mask_resulted)
cv2.waitKey(0)
cv2.destroyAllWindows()
hist=cv2.calcHist([img_GRAY],[0],mask,[256],[0,255])
plt.plot(hist.flatten(),color='teal')
plt.xlim([0,256])
plt.show()




img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_GRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([img_GRAY],[0],None,[256],[0,255])
img_eq=cv2.equalizeHist(img_GRAY)
hist_eq=cv2.calcHist([img_eq],[0],None,[256],[0,255])
fig,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].imshow(img_GRAY,cmap='gray')
axes[0,0].axis("off")
axes[0,1].imshow(img_eq,cmap='gray')
axes[0,1].axis("off")
axes[1,0].plot(hist.flatten(),color='black')
axes[1,0].set_xlim([0,256])
axes[1,1].plot(hist_eq.flatten(),color='black')
axes[1,1].set_xlim([0,256])
plt.tight_layout()
plt.show()



img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(img_LAB)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
L_enh=clahe.apply(L)
img_enh=cv2.merge([L_enh,A,B])
img_final_BGR=cv2.cvtColor(img_enh,cv2.COLOR_LAB2BGR)
img_final_RGB=cv2.cvtColor(img_final_BGR,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Ori_img")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(img_final_RGB)
plt.title("Edit_img with clahe")
plt.axis("off")
plt.tight_layout()
plt.show()








img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
img_LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(img_LAB)
L_enh=cv2.equalizeHist(L)
img_enh=cv2.merge([L_enh,A,B])
img_final_BGR=cv2.cvtColor(img_enh,cv2.COLOR_LAB2BGR)
img_final_RGB=cv2.cvtColor(img_final_BGR,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Ori_img")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(img_final_RGB)
plt.title("Edit_img with EqualizeHist")
plt.axis("off")
plt.tight_layout()
plt.show()



img_1=cv2.imread(r"C:\Users\Etijah\Desktop\bucharest-romania-november-closeup-side-view-red-porsche-gt-car-closeup-side-view-red-porsche-gt-car-parked-residential-344465962.jpg")
img_2=cv2.imread(r"C:\Users\Etijah\Desktop\H35_NORFOLK_II_EB_PORSCHE_MFS09853-Edit-Edit_16x9_6d2a5f83-a5c7-44ea-9cbc-18fcd3405576_1024x1024.jpg")
img_1_HSV=cv2.cvtColor(img_1,cv2.COLOR_BGR2HSV)
img_2_HSV=cv2.cvtColor(img_2,cv2.COLOR_BGR2HSV)
img_1_Hist=cv2.calcHist([img_1_HSV],[0,1],None,[50,60],[0,180,0,255])
img_2_Hist=cv2.calcHist([img_2_HSV],[0,1],None,[50,60],[0,180,0,255])
cv2.normalize(img_1_Hist,img_1_Hist)
cv2.normalize(img_2_Hist,img_2_Hist)
similarity=cv2.compareHist(img_1_Hist,img_2_Hist,cv2.HISTCMP_CORREL)
similarity_percent=similarity*100
print(f"The Similarit_percent :{similarity_percent}")













