import numpy as np
import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX
# points for 4 aruco points
arucoPoints = np.empty([4, 4, 2], dtype=int)
# points for homography
rectanglepoints = np.empty([4, 2], dtype=int)
cap = cv.VideoCapture(0)

#Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

# Initialize the detector parameters using default values
parameters =  cv.aruco.DetectorParameters_create()

# x1,y1 are top left, x2,y2 are bottom right
def get_grid_points(x1,y1,x2,y2):
    xpoints = np.empty([6], dtype=int)
    interval = int((x2-x1)/5)
    xpoints[0] = x1
    xpoints[1] = x1 + 1*interval
    xpoints[2] = x1 + 2*interval
    xpoints[3] = x1 + 3*interval
    xpoints[4] = x1 + 4*interval
    xpoints[5] = x2

    ypoints = np.empty([5], dtype=int)
    interval = int((y2-y1)/4)
    ypoints[0] = x1
    ypoints[1] = x1 + 1*interval
    ypoints[2] = x1 + 2*interval
    ypoints[3] = x1 + 3*interval
    ypoints[4] = x2

    coordinates = np.empty([5,4,2], dtype=int)
    for i in range(4):
        for j in range(5):
            coordinates[j][i][0] = xpoints[j]
            coordinates[j][i][1] = ypoints[i]

    print(coordinates)
    return coordinates

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    
    if(len(markerCorners) > 0):
        #print(markerCorners)
        print(len(markerIds))
        for i in range(len(markerIds)):
            
            for x in range(4):
                arucoPoints[i][x][0] = int(markerCorners[i][0][x][0])
                arucoPoints[i][x][1] = int(markerCorners[i][0][x][1])
            p1 = (arucoPoints[i][0][0], arucoPoints[i][0][1])
            p2 = (arucoPoints[i][1][0], arucoPoints[i][1][1])
            p3 = (arucoPoints[i][2][0], arucoPoints[i][2][1])
            p4 = (arucoPoints[i][3][0], arucoPoints[i][3][1])

            rectanglepoints[i][0] = arucoPoints[i][0][0]
            rectanglepoints[i][1] = arucoPoints[i][0][1]
            
            print(p1)
            print(p2)
            print(p3)
            print(p4)
            print('\n')
            
            frame = cv.line(frame,p1,p2,(255,0,0),5)
            frame = cv.line(frame,p2,p3,(255,0,0),5)
            frame = cv.line(frame,p3,p4,(255,0,0),5)
            frame = cv.line(frame,p4,p1,(255,0,0),5)

            id = str(int(markerIds[i]))
            frame = cv.putText(frame,id,p1, font, 1,(0,255,0),2,cv.LINE_AA)
    print(rectanglepoints)
    
    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
