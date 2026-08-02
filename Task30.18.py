import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 20.52.46.jpeg")
if img is None :
    raise FileNotFoundError("The Img not Loaded")

Palette_Names=["Red","Green","Blue","White","Black","Orange"]
Palette_bgr_List=[(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,0,0),(0,165,255)]
H,W=img.shape[:2]
Target_bgr=img[H//2,W//2]
def Closet_Color(Target_bgr,Palette_bgr_List):
    Distance=[]
    target_bgr_array=np.array(Target_bgr,dtype=np.uint8).reshape(1,1,3)
    target_lab=cv2.cvtColor(target_bgr_array,cv2.COLOR_BGR2LAB)[0,0]
    for i in Palette_bgr_List:
       Color_bgr_array=np.array(i,dtype=np.uint8).reshape(1,1,3)
       Palette_lab=cv2.cvtColor(Color_bgr_array,cv2.COLOR_BGR2LAB)[0,0]
       target_Lab1=target_lab.astype(float)
       Palette_Lab2=Palette_lab.astype(float)
       Delta_e=np.sqrt(np.sum((target_Lab1-Palette_Lab2)**2))
       Distance.append(Delta_e)
    Closest_index=np.argmin(Distance)
    Color_Name=Palette_Names[Closest_index]
    Closet_Distance=Distance[Closest_index]
    return Color_Name,Closet_Distance,Distance
Color_Name,Closet_Distance,Distance=Closet_Color(Target_bgr,Palette_bgr_List)
Result=list(zip(Palette_Names,Distance))
for rank, (name,Distance) in enumerate(Result,start=1):
    print(f"{rank}. {name}")
    print(f" Delta E={Distance:.1f}")
print("Closet Color:",Color_Name)
print("Delta E",Closet_Distance)


    


















