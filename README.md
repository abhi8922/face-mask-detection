# face-mask-detection
Face Mask Detection Project
What 
1.	It is a computer vision Application which can detect whether a person is wearing a mask or not and then store here data who are not wearing mask.
2.	 This Application can get data from various sources such as ip camera, Webcam, online video
3.	This is a web application which can be accessed over a local area network LAN
4.	Ip Camera is Wireless surveillance camera which can send footage to the WIFI network to which it is connected then the computer which contains the application is also connected with the same WIFi network, then we will be able to work on the footage data onto our project and detect mask.
Why
1.	His application can be used in the hospitals, research labs, polluted areas, Air borne pandemic, 
2.	 This kind of project shows the ML knowledge ,programming skills and web development wich can be very useful for resume,
3.	This kind of project shows the real world implantation, and real world problem solving with your knowledge
HOW:
1.	Backend:
a.	Connection with ip camera -> Open CV
b.	Face detection		-> Open CV
c.	Mask Detection		-> Keras
d.	Savedata			-> Open CV
2.	Front END:
Web Application - > Streamlit
1.	Overview And OpenCV
2.	Face detection
3.	Mask Detection and save data
4.	Frontend
OpenCV
Python Code Snippet:

Pip Install OpenCV
import cv2
vid=cv2.VideoCapture("camera url")
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        cv2.nameWindow("Flag",cv2.WINDOW_NORMAL)
        cv2.imshow("Flag",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

Face Detection
 
Python Code Snippet:

import cv2
facemodel=cv2.CascadeClassifier("face.xml")
vid=cv2.VideoCapture("camera url/video")
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)#[[x1,y1,l,w],[x1,y1,l,w],[x1,y1,w]]
        for (x,y,l,w) in faces:
            cv2.rectangle(frame,(x,y),(x+l,y+w),(77,255,0),5)      
        cv2.namedWindow("Abhi window",cv2.WINDOW_NORMAL)
        cv2.imshow("Abhi window",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()
Save Data
Python Code Snippet:
import cv2
facemodel=cv2.CascadeClassifier("face.xml")
vid=cv2.VideoCapture("camera url/video")
i=1
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)#[[x1,y1,l,w],[x1,y1,l,w],[x1,y1,w]]
        for (x,y,l,w) in faces:
            face_img=frame[y:y+w,x:x+1]
            path="data/"+str(i)+".jpg"
            i=i+1
            cv2.imwrite(path,face_img)
            cv2.rectangle(frame,(x,y),(x+l,y+w),(77,255,0),5)      
        cv2.namedWindow("Abhi window",cv2.WINDOW_NORMAL)
        cv2.imshow("Abhi window",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

Mask Detection
 
Python Code Snippet:
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
#to read an image detect mask and show it.
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')
img=cv2.imread("public.jpg")
face=facemodel.detectMultiScale(img)
for (x,y,h,w) in face:
    crop_face=img[y:y+w,x:x+h]
    cv2.imwrite('temp.jpg',crop_face)
    crop_face=load_img('temp.jpg',target_size=(150,150,3))
    crop_face=img_to_array(crop_face)
    crop_face=np.expand_dims(crop_face,axis=0)
    pred=maskmodel.predict(crop_face)[0][0]
    if pred==1:
        cv2.rectangle(img,(x,y),(x+h,y+w),(0,0,255),5)
    else:
        cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),5)
cv2.namedWindow("mywindow",cv2.WINDOW_NORMAL)
cv2.imshow("mywindow",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Frontend
Streamlit 
  
  
Python Code Snippet:
Pip install streamlit
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import tempfile
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')

st.title("Face Mask Detection System")
choice=st.sidebar.selectbox("My Menu",("Home","Image","Video","CAMERA"))
if(choice=="Home"):
    st.header("Welcome")
elif(choice=="Image"):
    file=st.file_uploader("Upload Image")
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(img)
        for (x,y,h,w) in face:
             crop_face=img[y:y+w,x:x+h]
             cv2.imwrite('temp.jpg',crop_face)
             crop_face=load_img('temp.jpg',target_size=(150,150,3))
             crop_face=img_to_array(crop_face)
             crop_face=np.expand_dims(crop_face,axis=0)
             pred=maskmodel.predict(crop_face)[0][0]
             if pred==1:
                 cv2.rectangle(img,(x,y),(x+h,y+w),(0,0,255),5)
             else:
                 cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),5)

        st.image(img,channels='BGR',width=400)
elif(choice=="Video"):
    file=st.file_uploader("Upload Video")
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
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
                window.image(frame,channels='BGR')

elif(choice=="CAMERA"):
    btn=st.button("Start Camera")
    window=st.empty()
    btn2=st.button('Stop Camera')
    if btn2:
        vid.close()
        st.experimental_rerun()
    if btn:
        vid=cv2.VideoCapture(0)
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
                window.image(frame,channels='BGR')
    
    



