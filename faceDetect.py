import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gender_net = cv2.dnn.readNetFromCaffe('./model/deploy_gender.prototxt', './model/gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('./model/deploy_age.prototxt', './model/age_net.caffemodel')

gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Detection with Gender and Age', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
