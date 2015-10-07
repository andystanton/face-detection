#!/usr/bin/env python3

import cv2
import sys
import numpy as np

if len(sys.argv) != 2:
    exit()

image = cv2.imread(sys.argv[1])
classifier_file = "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier("classifier/{}".format(classifier_file))

detected_faces = classifier.detectMultiScale(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    scaleFactor=1.2,
    minNeighbors=8,
    flags=cv2.CASCADE_SCALE_IMAGE,
    minSize=(40, 40))

for (x, y, width, height) in detected_faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)

cv2.imwrite("out.jpg", image)
