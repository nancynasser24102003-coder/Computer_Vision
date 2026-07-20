import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Etijah\Pictures\WhatsApp Image 2026-05-09 at 01.13.58.jpeg")
print(f"the of image : {img.shape}\nthe data type of image :{img.dtype}\nthe size of img : {img.size}")
H,W=img.shape[ :2]
Cx,Cy=W//2,H//2
print(f"Center Pixel (BGR):{img[Cy,Cx]}")
img_black=np.zeros((500,500),dtype=np.uint8)
H_black,W_black=img_black.shape
Cx_black,Cy_black=W_black//2,H_black//2
r=min(H_black,W_black)//10
y,x=np.ogrid[:H_black,:W_black]
mask=(x-Cx_black)**2+(y-Cy_black)**2<=r**2
img_black[mask]=255
cv2.imshow("Img",img_black)
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\output_1.jpg",img_black)
cv2.waitKey(0)
cv2.destroyAllWindows()
B,G,R=img[ :, : ,0],img[ : , : ,1] ,img[ : , : ,2]
plt.subplot(1,3,1)
plt.imshow(B,cmap='gray')
plt.title('B')
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(G,cmap='gray')
plt.title('G')
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(R,cmap='gray')
plt.title('R')
plt.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Task3_image.png")
plt.show()
img_f=img.astype(np.float32)/255.0
H_float,W_float=img_f.shape[:2]
Cx_float,Cy_float=W_float//2,H_float//2
print(f"Center Pixel in float type  :{img_f[Cy_float,Cy_float]}")
img_back=(img_f*255).astype(np.uint8)
print(f"Original center pixel: {img[Cy_float, Cx_float]}")
print(f"After converting back: {img_back[Cy_float, Cx_float]}")
















