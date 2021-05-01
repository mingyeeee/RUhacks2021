import numpy as np
import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX
# points for 4 aruco points
arucoPoints = np.empty([4, 4, 2], dtype=int)
cap = cv.VideoCapture(0)

def get_grid_points(x1, x2, y1, y2):
  # to do

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # if aruco marker is detected
    if(len(markerCorners) > 0):
        #print(markerCorners)
        print(len(markerIds))
        for i in range(len(markerIds)):
            
            # populating the numpy array with point values
            for x in range(4):
                arucoPoints[i][x][0] = int(markerCorners[i][0][x][0])
                arucoPoints[i][x][1] = int(markerCorners[i][0][x][1])
            # temporary variables for the purpose of convience 
            p1 = (arucoPoints[i][0][0], arucoPoints[i][0][1])
            p2 = (arucoPoints[i][1][0], arucoPoints[i][1][1])
            p3 = (arucoPoints[i][2][0], arucoPoints[i][2][1])
            p4 = (arucoPoints[i][3][0], arucoPoints[i][3][1])
            
            # debug
            print(p1)
            print(p2)
            print(p3)
            print(p4)
            print('\n')
            
            # drawing the lines
            frame = cv.line(frame,p1,p2,(255,0,0),5)
            frame = cv.line(frame,p2,p3,(255,0,0),5)
            frame = cv.line(frame,p3,p4,(255,0,0),5)
            frame = cv.line(frame,p4,p1,(255,0,0),5)
            
            # draw the ID on p1 
            id = str(int(markerIds[i]))
            frame = cv.putText(frame,id,p1, font, 1,(0,255,0),2,cv.LINE_AA)
            

    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
