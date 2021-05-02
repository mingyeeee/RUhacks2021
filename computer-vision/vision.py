import numpy as np
from imgsearch.aug_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
from time import time
import cv2

import os

# List of simulation videos
plant_vid = "./videos/plantcell.mov"
animal_vid ="./videos/animalcell.mov"
earth_vid = ""
chemistry_vid = "./videos/Presentation1.mov"

# List of splash screens
plant_screen = "./splashscreens/plant.png"
animal_screen = "./splashscreens/animal.png"
chemistry_screen = "./splashscreens/N2O4.png"

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

# start web cam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#-------------------------------------------------------------------------------
#             image scanning and sending to machine learning
#-------------------------------------------------------------------------------
while(True):
    frame = vs.read()
    cv2.putText(frame,"Please place your object in the box",(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Press q to start simulating",(60,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    
    # for get a 224 by 224 pixel cropped image
    height, width, channel = frame.shape
    y = int(height/2) - 112
    x = int(width/2) - 112
    crop_img = frame[y:y+224, x:x+224]
    #we can leave this out if we want to 
    cv2.rectangle(frame, (x,y), (x+224, y+ 224),(255,0,0), 3)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # saving image for machine learning
        cv2.imwrite("input_images\\done.jpg", crop_img)
        break
#-------------------------------------------------------------------------------
# Check and wait for a response from the machine learning on what to simulate
# -------------------------------------------------------------------------------
response = False
inputpath = 'input_images'

while(not response):
    frame = vs.read()
    cv2.putText(frame,"Analyzing setup...",(200,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    for entry in os.listdir(inputpath):
        if os.path.isfile(os.path.join(inputpath, entry)):
            if(entry == 'response.json'):
                os.remove("input_images\\response.json")
                response = True
    # Although theoretically unnecessary, the code breaks without :')
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

#-------------------------------------------------------------------------------
#                           Start of the simulations
# -------------------------------------------------------------------------------

if args["type"] == 1: 
    #standard
    # Change this
    video_path = animal_vid
    
    vf = cv2.VideoCapture(video_path)
    Q = deque(maxlen=128)
    (grabbed, source) = vf.read()
    Q.appendleft(source)
    
    # loop over the frames from the video stream
    while len(Q) > 0:
        frame = vs.read()
        # Change this
        cv2.putText(frame,"Animal cell microscope",(130,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,"simulation",(220,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        #frame = imutils.resize(frame, width=1000) #resize
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
       
    # loop over the frames from the video stream
    # Change this
    video_path = chemistry_vid

    vf = cv2.VideoCapture(video_path)
    Q = deque(maxlen=128)
    (grabbed, source) = vf.read()
    Q.appendleft(source)

    initialT = time()
    while len(Q) > 0:
        frame = vs.read()
        # delay for shaking the paper. once 5 seconds has elapsed the animation will play
        if(time() - initialT > 5):
            #print("Enough energy has been added to the system!")
            # Change this
            cv2.putText(frame,"Enough energy has been ",(110,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,"added to the system!",(130,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
            #frame = imutils.resize(frame, width=1000) #resize
            warped = find_and_warp(frame, source, cornerIDs=(1, 2, 3, 0), arucoDict=arucoDict, arucoParams=arucoParams, useCache=args["cache"] > 0)
            if warped is not None:
                frame = warped
                source = Q.popleft()
            if len(Q) != Q.maxlen:
                (grabbed, nextFrame) = vf.read()
                if grabbed:
                    Q.append(nextFrame)
        # message when simulation hasn't started
        else:
            #print("Shake the paper to add energy to the system!") 
            # Change this
            cv2.putText(frame,"Shake the paper to add energy!",(70,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,"N2O4 and NO2 equilibrium simulation",(24,190), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Formation of N2O4 gas",(55,220), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
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

    time.sleep(2.0)
    # loop over the frames from the video stream
    while len(Q) > 0:
        frame = vs.read()
        #frame = imutils.resize(frame, width=1000) #resize
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

