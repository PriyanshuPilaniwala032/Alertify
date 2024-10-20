import cv2
import os
import numpy as np
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO
import statistics

x_mouse=0
y_mouse=0


plot1 = plt.subplot2grid((2, 2), (0, 0))
plot2 = plt.subplot2grid((2, 2), (1, 0), rowspan=2)

model = YOLO("yolov5m6u.pt")


vid=cv2.VideoCapture(0) #webcam

#vid=cv2.VideoCapture("http://192.168.174.200:80/video")    #not working
#vid=cv2.VideoCapture("http://192.168.174.200:81/stream") #working
#vid=cv2.VideoCapture("final_test_case2.mp4")

height =int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width =int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps=vid.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter(("mytest.mp4"),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height),isColor=False)
out2=cv2.VideoWriter(("mytest_RGB.mp4"),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height),isColor=False)


def mouse_events(event,x,y,flags,param):

    if event==cv2.EVENT_MOUSEMOVE:

        global x_mouse
        global y_mouse

        x_mouse=x
        y_mouse=y

D1={} # dictionary to add details
num=0
L2=[]

while True:
    L1=[]   # list to collect the data of objects detected 
    num=num+1
    print(num)
    success,frame=vid.read()
    print("Start")
    frame2= cv2.resize(frame,(960,540))
    
    # object detection
    results = model.predict(frame2)
    
    result=results[0]
    #cv2.imshow("frame",result.plot())
    for box in result.boxes:
        label=result.names[box.cls[0].item()]
        if label=="person":
            L1.append(label)
        #print("object",label)
    no_of_per=len(L1)
    if num%2!=0:
        L2.append(no_of_per)
    else:
        
        print(L2)
        print(max(L2))
        D1[num]=max(L2)
        L2=[]

    # temperature part
    
    gray8=cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    gray82=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    data=gray8[y_mouse,x_mouse]
    print(data)

    cv2.circle(gray8, (x_mouse, y_mouse), 2, (0, 0, 0), -1)
    cv2.putText(gray8,"{0:.1f}".format(data),(x_mouse-80, y_mouse-15), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)

    result_temp= cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    result_temp2= cv2.applyColorMap(gray8, cv2.COLORMAP_MAGMA)
    screen1= np.concatenate((result_temp,frame2), axis=0)
    screen2= np.concatenate((result.plot(),result_temp2), axis=0)
    screen=np.concatenate((screen1,screen2),axis=1)

    cv2.imshow("out",screen)
    out.write(gray82)
    out2.write(frame)
    
    cv2.setMouseCallback("out",mouse_events)


    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
        
cv2.destroyAllWindows()
out.release()
print(D1)
print(D1.values())
key_lst=list(D1.keys())
key_values=list(D1.values())
print("Frames: ",key_lst)
key_new=[]
compare=[]
for i in key_lst:
    key_new.append(str(i))
    compare.append(i)
print("frames(str): ",key_new)
med=statistics.mean(key_values)

print("Mean value of person: ",med)
for i in compare:
    if D1[i] >= med:
        print(f"area {i} is high")
    elif D1[i]< med:
        print(f"area {i} is low")
value_lst=list(D1.values())
print(value_lst)

#print("HI")


plt.figure(figsize=(10,5))
#plt.bar(key_new,value_lst)
plt.plot(key_new,value_lst,marker='o')
plt.title('max person')
plt.ylabel('person')
plt.xlabel('no of frames')

plt.show()
#plt.savefig("output.jpg")
