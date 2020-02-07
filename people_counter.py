from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

class CentTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0


                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)


            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:

                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)


            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

#######################################################
class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]

		self.counted = False
#########################################################


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                               "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

if not args.get("input", False):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("Ouverture de la video ...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None


W = None
H = None

ct = CentTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}


totalFrames = 0
totalDown = 0
totalUp = 0

fps = FPS().start()

# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    if args["input"] is not None and frame is None:
        break

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)


    status = "Waiting"
    rects = []

    if totalFrames % 5 == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []


        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:

                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                #cv2.rectangle(frame, rect, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        for tracker in trackers:
            status = "Tracking"

            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 255), 4)#(0, 255, 255), 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    text = "Vers le haut: {}".format(totalUp)
    cv2.putText(frame, text, (10, H - ((20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    text = "Vers le bas: {}".format(totalDown)
    cv2.putText(frame, text, (10, H - ((320) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("RGB", rgb)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    totalFrames += 1
    fps.update()

fps.stop()

if not args.get("input", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
