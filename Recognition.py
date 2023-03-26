import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from PIL import Image
from imutils.video import VideoStream

protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
start_time = time.time()
recognizer = pickle.loads(open("output/recognizer", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

display_time =2
cap = VideoStream(0).start()
fps =FPS().start()
fc=0
FPS =0


def save_image_with_timestamp(frame):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())

    current_seconds = int(time.strftime("%S", time.gmtime()))

    if current_seconds - save_image_with_timestamp.last_seconds >= 8 or save_image_with_timestamp.last_seconds == 0:
        time.sleep(0.05)
        cv2.imwrite(f"unknown_faces/{current_time}.jpg", frame)
        save_image_with_timestamp.last_seconds = current_seconds


save_image_with_timestamp.last_seconds = 0


while True:
    frame = cap.read()
    fc+=1
    Time =time.time() -start_time
    if(Time)>=display_time:
        FPS =fc/(Time)
        fc=0
        start_time =time.time()
    FPS =int(FPS)
    fps_disp ="FPS"+str(FPS*2)[:5]
    frame =cv2.putText(frame,fps_disp,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    image = cv2.resize(frame, (300, 300))
    imageBlob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()


            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            if name =="unknown" and proba >0.93:
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)
                save_image_with_timestamp(frame)



            text = "{}: {:.2f}%".format(name, proba * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
