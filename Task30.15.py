import cv2
import numpy as np
import matplotlib.pyplot as plt 


img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-30 at 14.50.48.jpeg")
if img is None :
    raise FileNotFoundError("The image not loaded")

print(f"The Shape Of image : {img.shape} ")
Roi=img[120:500,400:850].copy()
Roi_ResizedLINEAR=cv2.resize(Roi,(400,400),cv2.INTER_LINEAR)
Roi_ResizeNEAREST=cv2.resize(Roi,(400,400),cv2.INTER_NEAREST)
Roi_ResizeCUBIC=cv2.resize(Roi,(400,400),cv2.INTER_CUBIC)

Roi_ResizedLINEARBatch60=Roi_ResizedLINEAR[80:140,160:220]
Roi_ResizeNEARESTBatch60=Roi_ResizeNEAREST[80:140,160:220]
Roi_ResizeCUBICBatch60=Roi_ResizeCUBIC[80:140,160:220]


fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax ,ch , name in zip(axes,[cv2.cvtColor(Roi_ResizedLINEAR,cv2.COLOR_BGR2RGB),cv2.cvtColor(Roi_ResizeNEAREST,cv2.COLOR_BGR2RGB),cv2.cvtColor(Roi_ResizeCUBIC,cv2.COLOR_BGR2RGB)],['Roi_ResizedLINEAR','Roi_ResizeNEAREST','Roi_ResizeCUBIC']):
    ax.imshow(ch)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Roi.png")
plt.show()



fig,axes=plt.subplots(1,3,figsize=(8,8))
for ax ,ch , name in zip(axes,[cv2.cvtColor(Roi_ResizedLINEARBatch60,cv2.COLOR_BGR2RGB),cv2.cvtColor(Roi_ResizeNEARESTBatch60,cv2.COLOR_BGR2RGB),cv2.cvtColor(Roi_ResizeCUBICBatch60,cv2.COLOR_BGR2RGB)],['Roi_ResizedLINEARBatch60','Roi_ResizeNEARESTBatch60','Roi_ResizeCUBICBatch60']):
    ax.imshow(ch)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\RoiBitch60.png")
plt.show()















