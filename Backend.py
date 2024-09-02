import cv2
facemodel=cv2.CascadeClassifier("face.xml")
vid=cv2.VideoCapture("http://192.168.29.223:8080/video")
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
