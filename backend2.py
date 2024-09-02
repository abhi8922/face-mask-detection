import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
#to read an image detect mask and show it.
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')
vid=cv2.VideoCapture("girls mask.mp4")
i=1
while(vid.isOpened()):
    flag,frame=vid.read()
    if flag:
        face=facemodel.detectMultiScale(frame)
        for (x,y,h,w) in face:
             crop_face1=frame[y:y+w,x:x+h]
             cv2.imwrite('temp.jpg',crop_face1)
             crop_face=load_img('temp.jpg',target_size=(150,150,3))
             crop_face=img_to_array(crop_face)
             crop_face=np.expand_dims(crop_face,axis=0)
             pred=maskmodel.predict(crop_face)[0][0]
             if pred==1:
                 cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),5)
                 path="C:/Projects/data/"+str(i)+".jpg"
                 cv2.imwrite(path,crop_face1)
                 i=i+1    
             else:
                 cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),5)
        cv2.namedWindow("mywindow",cv2.WINDOW_NORMAL)
        cv2.imshow("mywindow",frame)
        k=cv2.waitKey(5)
        if(k==ord('x')):
            break
    else:
         break
cv2.destroyAllWindows()