import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img_query=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 16.50.03.jpeg")
img_Candiate1=cv2.imread(r"C:\Users\Etijah\Pictures\WhatsApp Image 2026-07-31 at 16.50.02.jpeg")
img_Candiate2=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 16.50.02.jpeg")
img_Candiate3=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 16.50.46.jpeg")
img_Candiate4=cv2.imread(r"C:\Users\Etijah\Desktop\WhatsApp Image 2026-07-31 at 16.50.01.jpeg")

img_HSVQuery=cv2.cvtColor(img_query,cv2.COLOR_BGR2HSV)
img_HSVCandiate1=cv2.cvtColor(img_Candiate1,cv2.COLOR_BGR2HSV)
img_HSVCandiate2=cv2.cvtColor(img_Candiate2,cv2.COLOR_BGR2HSV)
img_HSVCandiate3=cv2.cvtColor(img_Candiate3,cv2.COLOR_BGR2HSV)
img_HSVCandiate4=cv2.cvtColor(img_Candiate4,cv2.COLOR_BGR2HSV)



Hist_Query=cv2.calcHist(img_HSVQuery,[0,1],None,[50,60],[0,180,0,256])
Hist_Candiate1=cv2.calcHist(img_HSVCandiate1,[0,1],None,[50,60],[0,180,0,256])
Hist_Candiate2=cv2.calcHist(img_HSVCandiate2,[0,1],None,[50,60],[0,180,0,256])
Hist_Candiate3=cv2.calcHist(img_HSVCandiate3,[0,1],None,[50,60],[0,180,0,256])
Hist_Candiate4=cv2.calcHist(img_HSVCandiate4,[0,1],None,[50,60],[0,180,0,256])






Query_Normalize=cv2.normalize(Hist_Query,Hist_Query,0,1,cv2.NORM_MINMAX)
Candiate1_Normalize=cv2.normalize(Hist_Candiate1,Hist_Candiate1,0,1,cv2.NORM_MINMAX)
Candiate2_Normalize=cv2.normalize(Hist_Candiate2,Hist_Candiate2,0,1,cv2.NORM_MINMAX)
Candiate3_Normalize=cv2.normalize(Hist_Candiate3,Hist_Candiate3,0,1,cv2.NORM_MINMAX)
Candiate4_Normalize=cv2.normalize(Hist_Candiate4,Hist_Candiate4,0,1,cv2.NORM_MINMAX)

Similarity_1=cv2.compareHist(Query_Normalize,Candiate1_Normalize,cv2.HISTCMP_CORREL)
Similarity_2=cv2.compareHist(Query_Normalize,Candiate2_Normalize,cv2.HISTCMP_CORREL)
Similarity_3=cv2.compareHist(Query_Normalize,Candiate3_Normalize,cv2.HISTCMP_CORREL)
Similarity_4=cv2.compareHist(Query_Normalize,Candiate4_Normalize,cv2.HISTCMP_CORREL)

Result_CORREL=[("Query vs Candiate1 ",Similarity_1),
               ("Query vs Candiate2 ",Similarity_2),
               ("Query vs Candiate3 ",Similarity_3),
               ("Query vs Candiate4 ",Similarity_4)
               ]


Ranked_CORREL=sorted(Result_CORREL, key=lambda kv:kv[1],reverse=True)
print("Ranked by Correlation (Upper More Similar)")

for i ,(name,score) in enumerate(Ranked_CORREL,start=1):
    print(f"{i} {name } {score:.4f}")




Similarity_1=cv2.compareHist(Query_Normalize,Candiate1_Normalize,cv2.HISTCMP_BHATTACHARYYA)
Similarity_2=cv2.compareHist(Query_Normalize,Candiate2_Normalize,cv2.HISTCMP_BHATTACHARYYA)
Similarity_3=cv2.compareHist(Query_Normalize,Candiate3_Normalize,cv2.HISTCMP_BHATTACHARYYA)
Similarity_4=cv2.compareHist(Query_Normalize,Candiate4_Normalize,cv2.HISTCMP_BHATTACHARYYA)

Result_BHATTACHARYYA=[("Query vs Candiate1", Similarity_1),
                      ("Query vs Candiate2", Similarity_2),
                      ("Query vs Candiate3", Similarity_3),
                      ("Query vs Candiate4", Similarity_4)
                      ]


Ranked_BHATTACHARYYA=sorted(Result_BHATTACHARYYA,key=lambda kv:kv[1])
print("Ranked by Bhattacharyya (Lower More Similar)")

for i,(name,score) in enumerate(Ranked_BHATTACHARYYA,start=1):
    print(f"{i} {name } {score:.4f}")



