import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')

index_to_emotion = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x-5, y-1), (x+w+5, y + h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_gray = clahe.apply(roi_gray)

        try:
            image = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(str(e))
            continue

        image = image.astype('float32') / 255
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
        img_fed = np.expand_dims(image, axis=0)

        scores = model.predict(img_fed)
        index = np.argmax(scores)
        emotion = index_to_emotion[index]

        print(f"Predicted scores: {scores}")
        print(f"Predicted emotion: {emotion}")

        frame = cv2.putText(frame, emotion, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), thickness=2)

    cv2.imshow('detection', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
