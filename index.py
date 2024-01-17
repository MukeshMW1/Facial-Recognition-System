import cv2
import os
import pickle
import numpy as np

video=cv2.VideoCapture(0)

faceDetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') 


face_data=[]

i=0
name=input("Enter your name")



if faceDetect.empty():
    print("Error: CascadeClassifier not loaded")
  
  


print("Current working directory:", os.getcwd())
cascade_path = 'data/haarcascade_frontalface_default.xml'
print("Cascade file exists:", os.path.exists(cascade_path))

faces = cv2.CascadeClassifier(cascade_path)

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        resize_img=cv2.resize(crop_img,(50,50))
        if len(face_data)<=100 and i%10==0:   
           face_data.append(resize_img)
        i=i+1
        cv2.putText(frame,str(len(face_data)) ,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50, 50, 255),1)
       
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow("Facial Attendance System",frame)
    
    k=cv2.waitKey(1)
    if k==ord('q') or len(face_data)==100:
      break
 
 
video.release()
cv2.destroyAllWindows() 


face_data=np.asarray(face_data).reshape(-1, 50 * 50 * 3)  # Adjust the reshape dimensions
# face_data=face_data.reshape(100,-1)

if 'name.pkl' not in os.listdir('data/'):
    names = [name] * (len(face_data))
    with open('data/names.pkl','wb') as f:        
        pickle.dump(names,f)

else:
    with open('data/names.pkl','rb') as f:
       names=pickle.load(f)
    names = names + [name] * (len(face_data) )
    with open('data/names.pkl','wb') as f:
       pickle.dump(names,f)


if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl','wb') as f:
      pickle.dump(face_data,f)

else:
    with open('data/face_data.pkl','rb') as f:
        faces=pickle.load(f)
        
        combined_face_data = np.concatenate([faces, face_data], axis=0)
        faces=np.append(faces,face_data,axis=0)
    with open('data/face_data.pkl','wb') as f:
         pickle.dump(faces,f)
