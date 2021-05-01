import cv2
import numpy as np

capture = cv2.VideoCapture(0)
image_path = './img/grid.png'
imgTarget = cv2.imread(image_path)

# import video capture
video_path = './video/Presentation1.mov'

myVid = cv2.VideoCapture(video_path)
# grab frame
ok, frame = myVid.read()
# resize our image video, so that the frame is the same size as the target image
# they must overlay exactly the same

h, w, c = imgTarget.shape
frame = cv2.resize(frame, (w, h))
detection = False
frameCounter = 0 #keeps the number of frames displayed in the video
# declare detector 
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# to see keypoints use draw function
imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


while True: 
    ok, videoFrame = capture.read()
    # define image augmentation
    imgAug = videoFrame.copy()
    # find decsriptor and keypoint of the second image that is the webcam
    kp2, des2 = orb.detectAndCompute(videoFrame, None)
    # videoFrame = cv2.drawKeypoints(videoFrame, kp2, None)

    if detection == False:
        # set the video back to frame 0
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else: 
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
                #repeat
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0    
        ok, frame = myVid.read()
        frame = cv2.resize(frame, (w, h))
    # compare both descriptors, using brute force matcher
    bf  = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k =2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    print(len(good)) # tells us how well the target is detected
    imgFeatures = cv2.drawMatches(imgTarget, kp1, videoFrame, kp2, good, None, flags=2)
    # homography
    if len(good) > 15: 
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) #loop and find each of the good matches
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0], [0,h], [w, h], [w, 0]]).reshape(-1,1,2) # size info of target image
        dst = cv2.perspectiveTransform(pts, matrix) # gives us the point where we have out target image
        img2 = cv2.polylines(videoFrame, [np.int32(dst)], True, (255, 0, 255), 3)
        imgWarp = cv2.warpPerspective(frame, matrix, (videoFrame.shape[1], videoFrame.shape[0]))
        # overlay:
        # create a mask
        maskNew = np.zeros((videoFrame.shape[0], videoFrame.shape[1]), np.uint8)
        # color the area where we find the image as white
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255, 255))
        # obtain the inverse of the previous line
        maskInverse = cv2.bitwise_not(maskNew)
        # colour the white region with our actual image
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInverse)
        # we have to add everything based on 'or' not 'and'
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

# use staking function to put all the images together
    # imgStacked = stackedImages(([videoFrame, imgWarp], [videoFrame, imgWarp]), 0.5)
    cv2.imshow('Image Augmentation', imgAug)
    # cv2.imshow('MaskNew', maskNew)
    # cv2.imshow('Image2', img2)
    # cv2.imshow('ImageFeatures', imgFeatures)
    # cv2.imshow('Image Target', imgTarget)
    # cv2.imshow('Video', frame)
    # cv2.imshow('Webcam', videoFrame)

    cv2.waitKey(1)
    frameCounter += 1 # increase the frameCounter by one each loop
