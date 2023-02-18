import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
face_data = []
dataset_path = './data/'

file_name = input("Enter the Person Name: ")

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.1, 3)
    #print(faces)
    faces = sorted(faces, key=lambda f:f[2]*f[3])

    # Pick the last face (because it is the largest face acc to area (f[2]*f[3]) )
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

        #Extrect (Crop out the request face)
        offset = 10
        face_selection = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_selection = cv2.resize(face_selection, (100,100))

        skip += 1
        if skip%10 ==0:
            face_data.append(face_selection)
            print(len(face_data))
        
    cv2.imshow("Video Frame", frame)
    cv2.imshow("Face Section", face_selection)

    key_pressed = cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy' , face_data)
print("data scussfully save at"+dataset_path+file_name + ".npy")

cap.release()
cv2.destroyAllWindows()