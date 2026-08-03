import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread(r"c:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 20.52.46.jpeg")
if img is None :
    raise FileNotFoundError("The Img Not Loaded")

H,W=img.shape[:2]
Cx=W//2
Cy=H//2
y,x=np.ogrid[:H,:W]
Distance=np.sqrt((x-Cx)**2+(y-Cy)**2)
Max_Distance=Distance.max()
Normalize_Distance=Distance/Max_Distance
Vignette_Mask=1-Normalize_Distance
Vignette_Mask=Vignette_Mask.astype(np.float32)
img_Float=img.astype(np.float32)
Vignette_Result=img_Float*Vignette_Mask[ : , : ,np.newaxis]
Vignette_Result=np.clip(Vignette_Result,0,255)
Vignette_Result=Vignette_Result.astype(np.uint8)



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(Vignette_Result,cv2.COLOR_BGR2RGB))
plt.title("Vignette Effect")
plt.axis("off")

plt.tight_layout()
plt.show()





















































