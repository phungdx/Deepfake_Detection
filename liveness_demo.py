# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='models/classifier.h5', type=str,
	help="path to trained model")
# ap.add_argument("-l", "--le", default='le.pickle', type=str,
# 	help="path to label encoder")
ap.add_argument("-d", "--detector",default='face_detector', type=str,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
# le = pickle.loads(open(args["le"], "rb").read())

threshold = 0.362891


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture("videos/test/fake/fake2.mp4")
time.sleep(2.0)

target = 3
counter = 0

# loop over the frames from the video stream
while True:
	if counter == target:
		counter = 0
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 600 pixels
		_,frame = vs.read()
		frame = imutils.resize(frame, width=600)
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the detected bounding box does fall outside the
				# dimensions of the frame
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)

				# extract the face ROI and then preproces it in the exact
				# same manner as our training data
				face = frame[startY:endY, startX:endX]
				face = cv2.resize(face, (160,160))
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)

				# pass the face ROI through the trained liveness detector
				# model to determine if the face is "real" or "fake"
				preds = model.predict(face)[0][0]
				# print(preds)
				# import sys
				# sys.exit()
				# j = np.max(preds)
				# label = le.classes_[j]
				if preds > threshold :
					label = 'real'

				else:
					label = 'fake'

				# draw the label and bounding box on the frame
				label = "{}: {:.4f}".format(label, preds)
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)

		# show the output frame and wait for a key press
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


	else:
		_ = vs.grab()
		counter+=1
		
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()