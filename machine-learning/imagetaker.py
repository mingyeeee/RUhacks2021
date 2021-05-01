import numpy as np
import cv2 as cv
from time import time
cap = cv.VideoCapture(0)
prev = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    height, width, channel = frame.shape

    y = int(height/2) - 112
    x = int(width/2) - 112
    crop_img = frame[y:y+224, x:x+224]
    
    if(time() - prev > 10):
        print("saving image")
        prev = time()
        filename = "image{time}.jpg".format(time=time())
        cv.imwrite(filename, crop_img)

    cv.rectangle(frame, (x,y), (x+224, y+ 224),(255,0,0), 3)

    # Display the resulting frame
    cv.imshow('crop',crop_img)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
