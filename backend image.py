import cv2
#to read an image and show it.
facemodel=cv2.CascadeClassifier("face.xml")
img=cv2.imread("woman2.png")
face=facemodel.detectMultiScale(img)
for (x,y,l,w) in face:
    cv2.rectangle(img,(x,y),(x+1,y+w),(0,0,0),4)
cv2.nameWindow("muwindow",cv2.WINDOW_NORMAL)
cv2.imshow("mywindow",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
