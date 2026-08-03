import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
if img is None :
    raise FileNotFoundError("The Img Not Loaded")



B,G,R=cv2.split(img)
B_float=B.astype(np.float32)
G_float=G.astype(np.float32)
R_float=R.astype(np.float32)




Gray_float=0.299*R_float+0.587*G_float+0.114*B_float

Gray_Round=np.round(Gray_float)

Gray_Manually=Gray_Round.astype(np.uint8)

Gray_Right=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

Gray_Difference=cv2.absdiff(Gray_Manually,Gray_Right)
Difference_Max=Gray_Difference.max()


print(f"Max Difference: {Difference_Max}")

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(Gray_Manually,cmap="gray")
plt.title("Manual Grayscale")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(Gray_Right,cmap="gray")
plt.title("OpenCV Grayscale")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(Gray_Difference,cmap="gray")
plt.title("Absolute Difference")
plt.axis("off")

plt.tight_layout()
plt.show()








































