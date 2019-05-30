#Users/macbookpro/anaconda3/envs/tensorflowenv
import numpy as np
import cv2
from keras.preprocessing import image

def get_input_type():
    type = int(input("""Chon nguon nhan dang hinh anh: 1: Video,2: Webcam
     Lua chon: """))
    return type

# detect face with opencv
def detect_emotion():
    face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
    type = get_input_type()
    if type == 2:
        cap = cv2.VideoCapture(0)
    elif type == 1:
        try:
            path = input("Duong dan den video: ")
            cap = cv2.VideoCapture(path)
        except Exception as e:
            print("Duong dan khong hop le")
            exit()
    else:
        print("Lua chon khong hop le")
        get_input_type()
    from keras.models import model_from_json
    model = model_from_json(open("model_structure.json", "r").read())
    model.load_weights('model_weights.h5') #load weights
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle around faces

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop faces
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48 x 48
            img_pixel = image.img_to_array(detected_face)
            img_pixel = np.expand_dims(img_pixel, axis=0)
            img_pixel /= 255

            prediction = model.predict(img_pixel)  # probabilities of 7 emotion

            max_index = np.argmax(prediction[0])  # find max index emotion

            emotion = emotions[max_index]
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Bam Q de thoat", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    detect_emotion()