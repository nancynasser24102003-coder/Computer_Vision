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




























