import cv2 as cv
from windowcapture import GameCapture
import numpy as np

#enter the name of the window here
window_name = "Grand Theft Auto V"

class_ids = []
confidences = []
boxes = []

frames = GameCapture(window_name)

nn = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers = nn.getLayerNames()
outputs = [layers[i[0] - 1] for i in nn.getUnconnectedOutLayers()]

frame = frames.get_frame()
height, width, channels = frame.shape
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    frame = frames.get_frame()

    blob = cv.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    nn.setInput(blob)
    fOuts = nn.forward(outputs)

    for out in fOuts:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv.QT_FONT_NORMAL
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label, (x, y + 30), font, 3, color, 3)

    cv.imshow("screen", frame)

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
