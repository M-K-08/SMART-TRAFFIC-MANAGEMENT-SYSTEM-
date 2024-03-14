import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from keras.models import load_model
length=0


def length_t(id):
    global length
    if id=='car':
        length+=2
    elif id=='truck':
        length+=5
    elif id=='bus':
        length+=5
    return length


num_of_lanes=2
emergency=load_model('emergencyve')
model=YOLO('yolov8n.pt')

cap=cv2.VideoCapture('cars.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0
area=[(200,380),(950,380),(950,410),(200,410)]
tlx1=200
tly1=300
brx2=950
bry2=410
target_classes = ["car", "truck", "motorcycle"]
tracker=Tracker()
area_c=set()
eclass=["not emv","yes emv"]
ans={}
while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(1020,500))
    
    # cv2.imshow('image window', frame[tly1:bry2,tlx1:brx2])
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    list=[]
    for index,row in px.iterrows():
        # print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if x1<tlx1 and y1<tly1 and x2>brx2 and y2>bry2 :
            continue 
        if c in target_classes:
            list.append([x1,y1,x2,y2])

    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        new_results=model.predict(frame[y3:y4,x3:x4])
        z=new_results[0].boxes.data
        new_px=pd.DataFrame(z).astype("float")
        c="nn"
        d=0
        for index,row in px.iterrows():
            d=int(row[5])
            c=class_list[d]
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if c not in target_classes:
            continue
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255),2)
            cv2.putText(frame,str(c),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            area_c.add(id)
            section_image = frame[y3:y4, x3:x4]
            section_image = cv2.cvtColor(section_image, cv2.COLOR_BGR2RGB)
            section_image=cv2.resize(section_image,(224,224))
            cv2.imshow('read window', section_image)
            # section_image = np.array(section_image)
            # section_image = np.expand_dims(section_image, axis=0)
            # section_image=section_image/255.0
            # ans=emergency.predict(section_image)
            # pred = np.argmax(ans)
            # print(eclass[pred])
            if c not in ans:
                ans[c]=1
            else:
                ans[c]+=1
            
    carcount=len(area_c)
    # total=length_t(c)
    cv2.putText(frame,str(carcount),(50,50),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),2)
    # cv2.putText(frame,str(total),(100,100),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),2)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
print(ans)
cap.release()
cv2.destroyAllWindows()





