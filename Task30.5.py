import cv2
import numpy as np 
import matplotlib.pyplot as plt 

try:
    img_BGR=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshot_20260722_232311_ChatGPT.jpg")
    if img_BGR is None:
        raise FileNotFoundError("The Image not Loaded")
except FileNotFoundError as error:
    print(error)

img_RGB=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(img_BGR)
plt.title("Wrong (BGR fed as RGB)")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(img_RGB)
plt.title("Correct (converted to RGB)")
plt.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Task5.png")
plt.show()











