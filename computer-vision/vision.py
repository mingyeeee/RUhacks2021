import numpy as np
from imgsearch.aug_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
import time
import cv2

CACHED_REF_PTS = None
#if statments while loop for each 

# type1 run of the bath
# type 2, wait because you have to initiate so you shake
# type 3, switch frame of movement...threshold that is when it switch the frame.

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", type=str, required=True, help="path to input video file for augmented reality")
ap.add_argument("-c", "--cache", type=int, default=-1, help="whether or not to use reference points cache")
ap.add_argument("-t", "--type", type=int, default=1, help="type of simulation, 1 2 or 3" )
args = vars(ap.parse_args())
print("[INFO] initializing marker detector...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
print("[INFO] accessing video stream...")
if args["type"] == 1: 
    #standard
    video_path = "./videos/plantcell.mov"
    vf = cv2.VideoCapture(video_path)
    Q = deque(maxlen=128)
    (grabbed, source) = vf.read()
    Q.appendleft(source)
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # loop over the frames from the video stream
    while len(Q) > 0:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000) #resize
        warped = find_and_warp(frame, source, cornerIDs=(1, 2, 3, 0), arucoDict=arucoDict, arucoParams=arucoParams, useCache=args["cache"] > 0)
        if warped is not None:
            frame = warped
            source = Q.popleft()
        if len(Q) != Q.maxlen:
            (grabbed, nextFrame) = vf.read()
            if grabbed:
                Q.append(nextFrame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
#kinetic energy
elif args["type"] == 2: 
    video_path = "./videos/Presentation1.mov"
    vf = cv2.VideoCapture(video_path)
    Q = deque(maxlen=128)
    (grabbed, source) = vf.read()
    Q.appendleft(source)
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
   # initialT = time()   
    # loop over the frames from the video stream
    while len(Q) > 0:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000) #resize
        warped = find_and_warp(frame, source, cornerIDs=(1, 2, 3, 0), arucoDict=arucoDict, arucoParams=arucoParams, useCache=args["cache"] > 0)
        if warped is not None:
            frame = warped
            source = Q.popleft()
        if len(Q) != Q.maxlen:
            (grabbed, nextFrame) = vf.read()
            if grabbed:
                Q.append(nextFrame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
elif args["type"] == 3: 
    # imgOn = "./img/oncircuit.png"
    # imgOff = "./img/offcircuit.png"

    # onSrc  = cv2.imread(imgOn)
    # offSrc  = cv2.imread(imgOff)
    video_path = "./videos/Presentation1.mov"
    vf = cv2.VideoCapture(video_path)
    Q = deque(maxlen=128)
    (grabbed, source) = vf.read()
    Q.appendleft(source)
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # loop over the frames from the video stream
    while len(Q) > 0:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000) #resize
        warped = find_and_warp(frame, source, cornerIDs=(1, 2, 3, 0), arucoDict=arucoDict, arucoParams=arucoParams, useCache=args["cache"] > 0)
        if warped is not None:
            frame = warped
            source = Q.popleft()
        if len(Q) != Q.maxlen:
            (grabbed, nextFrame) = vf.read()
            if grabbed:
                Q.append(nextFrame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()

