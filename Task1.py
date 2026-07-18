import cv2
import numpy as np
import matplotlib.pyplot as plt 




img=cv2.imread(r"C:\Users\Etijah\Pictures\WhatsApp Image 2026-05-09 at 01.13.58.jpeg")
img_f=img.astype(np.float32)/255.0
img_processed=img_f*1.2
img_clip=np.clip(img_processed,0.0,1.0)
img_final=(img_clip*255).astype(np.uint8)
print(f"The Data Type img_f is:{img_f.dtype} ,Type of img is {img.dtype} ")
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\Bright_image.png",img_final)
cv2.imshow("Origina_Image",img)
cv2.imshow("Bright_Imaged",img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()





img_1=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshots\Screenshot 2025-12-16 190300.png")
img_2=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshots\Screenshot 2025-12-16 190510.png")
img_3=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshots\Screenshot 2026-04-20 010531.png")
HeightofImageOne,WidthofImageOne,NumofArrayofImageOne=img_1.shape
HeightofImageTwo,WidthofImageTwo,NumofArrayofImageTwo=img_2.shape
HeightofImageThree,WidthofImageThree,NumofArrayofImageThree=img_3.shape
TotalPixelsCountofImageOne=HeightofImageOne*WidthofImageOne
TotalPixelsCountofImageTwo=HeightofImageTwo*WidthofImageTwo
TotalPixelsCountofImageThree=HeightofImageThree*WidthofImageThree
CxofImageOne,CyofImageOne=HeightofImageOne//2,WidthofImageOne//2
CxofImageTwo,CyofImageTwo=HeightofImageTwo//2,WidthofImageTwo//2
CxofImageThree,CyofImageThree=HeightofImageThree//2,WidthofImageThree//2
Center_Pixel_Img1=img_1[CxofImageOne,CyofImageOne]
Center_Pixel_Img2=img_2[CxofImageTwo,CyofImageTwo]
Center_Pixel_Img3=img_3[CxofImageThree,CyofImageThree]

if img_1 is not None and img_2 is not None and img_3 is not None :
    print("All Images are loaded")
    print("#"*50)
    cv2.imshow("Image1",img_1)
    cv2.imshow("Image2",img_2)
    cv2.imshow("Image3",img_3)
    print(f"The Shape og Image 1 is : {img_1.shape} \nThe Shape og Image 2 is : {img_2.shape} \nThe Shape og Image 3 is : {img_3.shape} ")
    print("#"*50)
    print(f"The Data Type of Image 1 is :{img_1.dtype} \nThe Data Type of Image 2 is :{img_2.dtype} \nThe Data Type of Image 3 is :{img_3.dtype}")
    print("#"*50)
    print(f"The Image 1 Type in Memory is {type(img_1)} \nThe Image 2 Type in Memory is {type(img_2)} \nThe Image 3 Type in Memory is {type(img_3)} ")
    print("#"*50)
    print(f"The Total Pixels Count of Image One :{TotalPixelsCountofImageOne}\nThe Total Pixels Count of Image Two :{TotalPixelsCountofImageTwo}\nThe Total Pixels Count of Image Three :{TotalPixelsCountofImageThree}")
    print("#"*50)
    print(f"The Pixels Values of The Image one in the Center  : {Center_Pixel_Img1} \nThe Pixels Values of The Image Two in the Center{Center_Pixel_Img2} \nThe Pixels Values of The Image Three in the Center: {Center_Pixel_Img3}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



else :
    print("Error")    



img1=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshots\Screenshot 2026-05-06 083229.png")
Blue,Green,Red=cv2.split(img1)
cv2.imshow("Blue",Blue)
cv2.imshow("Green",Green)
cv2.imshow("Red",Red)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\Etijah\Desktop\CVImage\Gray_image1.png",Blue)


img2=cv2.imread(r"C:\Users\Etijah\Pictures\Screenshots\Screenshot 2026-05-03 144722.png")
plt.subplot(1,3,1)
plt.imshow(Blue,cmap='gray')
plt.title("Blue")
plt.subplot(1,3,2)
plt.imshow(Green,cmap='gray')
plt.title("Green")
plt.subplot(1,3,3)
plt.imshow(Red,cmap='gray')
plt.title("Red")
plt.savefig(r"C:\Users\Etijah\Desktop\CVImage\Gray_image2.png")
plt.show()


