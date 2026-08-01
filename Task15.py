import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-29 at 20.36.59.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

Sobelx=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
Sobely=cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
Edges_Sobel=cv2.magnitude(Sobelx,Sobely)
Edges_Sobel_u8=cv2.convertScaleAbs(Edges_Sobel)
cv2.imshow("Edges with Sobel",Edges_Sobel_u8)
cv2.waitKey(0)
cv2.destroyAllWindows()


Edges_Laplacian=cv2.Laplacian(img_gray,cv2.CV_64F,ksize=3)
Edges_Laplacian_u8=np.uint8(np.absolute(Edges_Laplacian))


Blurred_gray=cv2.GaussianBlur(img_gray,(5,5),0)
Edges_Laplacian_Blurred=cv2.Laplacian(Blurred_gray,cv2.CV_64F,ksize=3)
Edges_Laplacian__Blurred_u8=np.uint8(np.absolute(Edges_Laplacian_Blurred))

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(Edges_Laplacian_u8, cmap='gray')
plt.title("Laplacian - Original Gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(Edges_Laplacian__Blurred_u8, cmap='gray')
plt.title("Laplacian - After Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()


Edges_Canny=cv2.Canny(img_gray,50,150,apertureSize=3,L2gradient=True)

Edges_Canny_Blurred=cv2.Canny(Blurred_gray,50,150,apertureSize=3,L2gradient=True)




Edge_Pixel_Count = np.count_nonzero(Edges_Canny)

print(f"Canny Edge Pixel Count of Canny: {Edge_Pixel_Count}")


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(Edges_Canny, cmap='gray')
plt.title("Canny - Original Gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(Edges_Canny_Blurred, cmap='gray')
plt.title("Canny - After Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(Edges_Sobel_u8, cmap='gray')
plt.title("Sobel Magnitude")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(Edges_Laplacian_u8, cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(Edges_Laplacian__Blurred_u8, cmap='gray')
plt.title("Laplacian + Gaussian Blur")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(Edges_Canny_Blurred, cmap='gray')
plt.title("Canny + Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()


Blur_K=(5,5)
Canny_Low=500
Canny_High=600
Edges_Color=(0,255,0)
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Can not Open Camera")

while True :

    ret,frame=cap.read()

    if not ret :
        break

    Blurred=cv2.GaussianBlur(frame,Blur_K,0)
    Gray_Blurred=cv2.cvtColor(Blurred,cv2.COLOR_BGR2GRAY)
    Edges=cv2.Canny(Gray_Blurred,Canny_Low,Canny_High,apertureSize=5,L2gradient=True)
    output=frame.copy()
    output[Edges==255]=Edges_Color
    Edges_BGR=cv2.cvtColor(Edges,cv2.COLOR_GRAY2BGR)
    Combined=np.hstack([frame,Edges_BGR,output])
    Combined=cv2.resize(Combined,(1500,500))
    cv2.imshow("Edges Layer",Combined)



    key=cv2.waitKey(1) & 0xFF


    if key==ord('c'):
      break
    if key==ord('s'):
        cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\Video_Capture.png",Combined)



cap.release()
cv2.destroyAllWindows()





















