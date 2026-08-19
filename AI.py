import cv2 
import numpy as  np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

model=tf.keras.models.load_model("keras_model.h5")
with open("labels.txt","r") as f:
    labels=[line.strip() for line in f]

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    img=cv2.resize(frame,(224,224))
    img=np.astype(np.float32)
    img=(img/127.5)-1
    img=np.expand_dims(img,axis=0)
    prediction=model.perdict(img,verbose=0)
    class_id=np.argmax( prediction[0])
    label=labels[class_id]
    confidence=prediction[0][class_id]
    cv2.putText(frame,f"{label}: {confidence*100:.1f}%",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("AI Classification",frame)
    if cv2.waitKey(1) & 0xFF==ord("c"):
        break
cap.release()
cv2.destroyAllWindows()




















































