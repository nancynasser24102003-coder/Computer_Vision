import cv2
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\659349064_18576944251041430_2870406156677966472_n-1.jpg")
if img is None :
    raise FileNotFoundError("The Image not Loaded")

img_Original_Clean=img.copy()
img_copy=img.copy()
Roi=img[100:200,100:200]
Roi[:,:]=[255,0,0]
Roi_copy=img_copy[100:200,100:200].copy()
Roi_copy[:,:]=[255,0,0]




plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Buggy Original was Mutated")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB))
plt.title("Fixed: Original untouched ")
plt.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\task9.png")
plt.show()



if np.array_equal(img_Original_Clean,img):
    print("True")
else:
    print("False")


if np.array_equal(img_Original_Clean,img_copy):
    print("True")
else:
    print("False")














