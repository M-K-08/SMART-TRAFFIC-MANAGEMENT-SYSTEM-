!pip install --upgrade pip setuptools
!pip uninstall lida
!pip install fastapi kaleido python-multipart uvicorn
!pip install lida
!pip install wget
!pip install --upgrade opencv-python


import wget

# URL of the file to download
url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p5.cfg"

# Destination path to save the file
config_path = "yolov4.cfg"

# Download the file
wget.download(url, config_path)



url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.weights"

# Destination path to save the file
weights_path = "yolov4.weights"

# Download the file
wget.download(url, weights_path)



url = "https://raw.githubusercontent.com/taipingeric/yolo-v4-tf.keras/master/class_names/coco_classes.txt"

# Destination path to save the file
classes_path = "classes.txt"

# Download the file
wget.download(url, classes_path)



import numpy as np
import imutils
import cv2

input_name="input_video.mp4"
output_name="output_video.avi"
inputVideoPath =f'./{input_name}'
outputVideoPath = f'./{output_name}'
yoloWeightsPath = f'./{weights_path}'
yoloConfigPath = f'./{config_path}'
detectionProbabilityThresh = 0.5
nonMaximaSuppression = 0.3

labelsPath = f'./{classes_path}'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNet(yoloConfigPath, yoloWeightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(inputVideoPath)
writer = None
(W, H) = (None,None)

try:
    prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))

except:
    total = -1
    print('Frames could not be determined')

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None and H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > detectionProbabilityThresh:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, detectionProbabilityThresh, nonMaximaSuppression)
    # ensure at least one detection exists
    print(len(idxs))       #Number of Vehicles
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputVideoPath, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

