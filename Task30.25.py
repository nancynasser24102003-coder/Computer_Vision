import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
Foulder_Path=r"C:\Users\Etijah\Desktop\Images"
Files=os.listdir(Foulder_Path)
print(Files)
Image_Files=[]

for File in Files:
    if File.lower().endswith((".jpg",".jpeg",".png")):
        Image_Files.append(File)
Images=[]
for File in Image_Files:
    Img_path=os.path.join(Foulder_Path,File)
    img=cv2.imread(Img_path)
    if img is not None:
        Images.append(img)

Number_OF_Images=len(Images)

if Number_OF_Images <4:
    raise ValueError("The folder must contain at least 4 images")

Columns=math.ceil(math.sqrt(Number_OF_Images))
Rows=math.ceil(Number_OF_Images/Columns)

Width=200
Height=200
Canvas_Width=Columns*(Width+10)
Canvas_Height=Rows*(Height+10)
Canvas=np.zeros((Canvas_Height,Canvas_Width,3),dtype=np.uint8)
Canvas[:]=(0,255,0)

for i ,img in enumerate(Images):
    Row=i//Columns
    Column=i%Columns
    Resized_Image=cv2.resize(img,(Height,Width),interpolation=cv2.INTER_AREA)
    Y=Row*(Height+10)
    X=Column*(Width+10)
    Canvas[Y:Y+Height,X:X+Width]=Resized_Image


plt.imshow(cv2.cvtColor(Canvas,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()







































































