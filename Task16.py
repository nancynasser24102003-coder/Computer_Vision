import cv2
import numpy as np 
import matplotlib.pyplot as plt 


img=cv2.imread(r"C:\Users\Etijah\Desktop\file_00000000ac2c8243a0c65deed7cda673[1].png")
if img is None :
    raise FileNotFoundError("The Image Not Loaded")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
T=200
T,Binary_Mask=cv2.threshold(img_gray,T,255,cv2.THRESH_BINARY_INV)
Kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
Opened=cv2.morphologyEx(Binary_Mask,cv2.MORPH_OPEN,Kernel)
Clean_Mask=cv2.morphologyEx(Opened,cv2.MORPH_CLOSE,Kernel)
Contours,Hierarchy=cv2.findContours(Clean_Mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(f"{'Index':<6} {'Area':<10} {'Perimeter':<12} {'Centroid':<15} {'Bounding Rect':<300} {'Aspect':<10} {'Extent':<10} {'Solidity':<10}")
MIN_AREA=500
Valid_Contours=[]
for c in Contours:
    Area=cv2.contourArea(c)
    if Area>=MIN_AREA:
      Valid_Contours.append(c)
print(f"Num of Valid Contours: {len(Valid_Contours)}")
# cv2.drawContours(img,Valid_Contours,-1,(0,255,255),2)
for i , c in enumerate(Valid_Contours):
   Area=cv2.contourArea(c)
   Perimeter=cv2.arcLength(c,True)
   epsilon=0.02 * cv2.arcLength(c,True)
   Approx=cv2.approxPolyDP(c,epsilon,True)
   Vertices=len(Approx)
   cv2.polylines(img,[Approx],True,(0,255,255),2)
   M=cv2.moments(c)
   if M["m00"]!=0:
     Cx=int(M["m10"]/M["m00"])
     Cy=int(M["m01"]/M["m00"])
   cv2.circle(img,(Cx,Cy),5,(255,255,255),-1)
   x,y,w,h=cv2.boundingRect(c)
   Aspect_Ratio=w/h
   Extent=Area/(w*h)
   Hull=cv2.convexHull(c)
   Hull_Area=cv2.contourArea(Hull)
   Solidity=Area/Hull_Area
   print(f"{i:<6} {Area:<10.2f} {Perimeter:<12.2f} {str((Cx,Cy)):<15} {str((x,y,w,h)):<30} {Aspect_Ratio:<10.2f} {Extent:<10.2f} {Solidity:<10.2f}")

   if Vertices==3:
        shape="Triangle"
        
   elif Vertices==4:
        if 0.9<=Aspect_Ratio <=1.1:
            shape="Square" 
            
        else:
            shape="Rectangle"
           
   elif Vertices==5:
        shape="pentagon"
       
   elif Vertices==6:
        shape="Hexagon"
       
   elif Vertices > 6:
        shape="Circle"
   print(f"Contour {i} :Vertices = {Vertices}")
   cv2.putText(img,shape,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
   cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),2)


cv2.imshow("Contours",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
