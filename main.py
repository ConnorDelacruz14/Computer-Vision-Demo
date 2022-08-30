import cv2 as cv
import sys
import numpy as np

# Look for camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

capture = cv.VideoCapture(s)
threshold = 0.5 # 50% threshhold to detect object
nms_threshold = 0.2 #  Used for deciding between objects


classNames = []
# Relative Paths for now
classFile = 'coco.names' 
config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

with open(classFile, 'rt') as f:
    # put all the classes into a list (objects)
    classNames = f.read().rstrip('\n').split('\n')

# Detection
net = cv.dnn_DetectionModel(weights, config)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = capture.read()
    #If 50% sure the detected object is an instance of an
    # object then show detection
    classIDs, confs, bbox = net.detect(img,confThreshold=threshold)
    # print object, bounding box, and confidence level
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)

    for i, confidence in zip(indices, confs):
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x+w, h+y), color=(0, 255,0), thickness=2)
        cv.putText(img, classNames[classIDs[i]-1].upper(), (box[0]+10, box[1]+30), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv.putText(img, str(confidence * 100)[:4] + "%", (box[0]+150, box[1]+30), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

    cv.imshow("Output", img)
    cv.waitKey(1)
