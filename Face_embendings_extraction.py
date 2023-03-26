from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
# import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
from imutils.video import FPS
from imutils.video import VideoStream


protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
imagePaths = list(paths.list_images("dataset"))
knownEmbeddings = []
knownNames = []
total = 0
test=[]
for (i, imagePath) in enumerate(imagePaths):
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlob)
	detections = detector.forward()
	test.append(detections)
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			rectangle = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x1, y1, x2, y2) = rectangle.astype("int")
			face = image[y1:y2, x1:x2]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			faceblob = cv2.dnn.blobFromImage(face, 1.0 / 255,
											 (96, 96), (0, 0, 0), swapRB=True, crop=False)

			embedder.setInput(faceblob)
			vector = embedder.forward()
			knownNames.append(name)
			knownEmbeddings.append(vector.flatten())
			total += 1

# conf= np.argmax(test[1][0, 0, :, 2])
# confidence = test[1][0, 0, conf, 2]
# print(confidence)
# print(vector)
# test_rectangle = test[0][0, 0, conf, 3:7] * np.array([w, h, w, h])


print(total)
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()


data = pickle.loads(open("output/embeddings.pickle", "rb").read())
le = LabelEncoder()
labels = le.fit_transform(data["names"])

recognizer = SVC(C=1.0, kernel="rbf", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("output/recognizer", "wb")
f.write(pickle.dumps(recognizer))
f.close()



