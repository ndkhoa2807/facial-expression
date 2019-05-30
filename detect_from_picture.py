#!/Users/macbookpro/anaconda3/envs/tensorflowenv/bin/python3.6
#-*- coding:utf-8 -*-
import os
import argparse
import cv2
import tensorflow as tf
from tensorflow.python.util import deprecation
import keras
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import numpy as np

# Hide tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False

with open("model_structure.json", "r") as f:
    structure = f.read()
    model = model_from_json(structure)

model.load_weights('model_weights.h5')

emotions = ('gian du', 'ghe tom', 'so hai', 'vui ve', 'buon', 'bat ngo', 'binh thuong')


def get_faces(img_path):
    while True:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        img = cv2.imread(img_path)
        if img is None:
            print("Khong tim thay anh")
            break
        else:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.1, 5)
            if faces is None:
                print("Khong tim thay mat trong anh")
                return 0

            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    detected_face = cv2.resize(detected_face, (48, 48))
                    img_pixel = img_to_array(detected_face)
                    img_pixel = np.expand_dims(img_pixel, axis=0)
                    prediction = model.predict(img_pixel)
                    max_index = np.argmax(prediction[0])
                    cv2.putText(img, emotions[max_index], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Nhan Q de thoat', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = input("Duong dan den anh: ")
    get_faces(path)