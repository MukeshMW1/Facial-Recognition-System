from sklearn.neighbors import KNeighborsClassifier


import cv2
import os
import pickle
import numpy as np
import csv
import time
from datetime import datetime

from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch("SAPI.SpVoice")
    speak.Speak(str1)
    
    
    
video=cv2.VideoCapture(0)

faceDetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') 


with open('data/names.pkl','rb') as f:
       LABELS=pickle.load(f)
       
with open('data/face_data.pkl','rb') as f:
            FACES=pickle.load(f)


print("Number of samples in FACES:", len(FACES))
print("Number of samples in LABELS:", len(LABELS))


if len(FACES) != len(LABELS):
    print("Error: Inconsistent sizes of FACES and LABELS")
else:
    print("Sizes are consistent. Fitting KNN...")
if faceDetect.empty():
    print("Error: CascadeClassifier not loaded")
  
  


print("Current working directory:", os.getcwd())
cascade_path = 'data/haarcascade_frontalface_default.xml'
print("Cascade file exists:", os.path.exists(cascade_path))


faces = cv2.CascadeClassifier(cascade_path)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

# COLS_NAMES=['NAMES','TIME','DATE']
    

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
#     faces=faceDetect.detectMultiScale(gray, 1.3, 5)
#     for(x,y,w,h) in faces:
#         crop_img=frame[y:y+h,x:x+w,:]
#         resize_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
#         output= knn.predict(resize_img)
#         ts=time.time()
#         cv2.putText(frame,output[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#         timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_"+ date + ".csv")
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#         attendance=[str(output[0]),str(timestamp),str(date)]
        
#     cv2.imshow("frame",frame)
    
#     k=cv2.waitKey(1)
#     if k==ord('o'): 
#         speak("Attendance Taken...")
#         time.sleep(3)
#         if exist:
#               with open("Attendance/Attendance_"+ date + ".csv" , "a") as csvfile:
#                writer= csv.writer(csvfile)
#                writer.writerow(attendance) 
#         else:
#             with open("Attendance/Attendance_"+ date + ".csv" , "a") as csvfile:
#                writer= csv.writer(csvfile)
#                writer.writerow(COLS_NAMES)
#                writer.writerow(attendance) 
#             csvfile.close()                        
#     if k==ord('q'):
#       break
 
 
# video.release()
# cv2.destroyAllWindows() 


