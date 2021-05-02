

![AugmentEd](./computer-vision/img/logo-removebg-preview.jpg)

With AugmentEd, our goal is to provide students, and institutions with an accessible, affordable, and interactive way of learning.

![](./computer-vision/img/animation.png)

## What is AugmentedEd?

A fun, interactive educational tool that projects virtual simulations of science experiments onto your sheet of paper using only your camera, a printer, and paper!



## How does it work?

- Draw a picture of the experiment you want to visualize on paper. 
- A pre-trained machine learning model identifiess the drawing based on a trained MobileNetv2 Network
- A virtual 3D model of the experiment is rendered and animated 
- The animation is projected onto your piece of paper


![](./computer-vision/img/signup_2.png)


![](./computer-vision/img/example.png)



## Backend

```
AugmentEd ────────computer-vision
                │           ├─./img
                │           ├─./imgsearch   
                │           │         │─ __init__.py
                │           │         │─ aug_reality.py
                │           │         │─ __pycache__
                │           │
                │           │
                │           └─ ./sim model
                │           │         │  
                │           │         │─./saved_model
                │           │         │─ label.txt
                │           │
                │           │
                │           └─ ./videos
                │           │      
                │           │       
                │           │             
                │           │    
                │           │─ simulation_classifier.py  
                │           └─  vision.py   
                │ 
                │──── machine-learning
                │           │
                │           │
                │           └─ ./sim model
                │           │         │  
                │           │         │─./saved_model
                │           │         └─  label.txt
                │           │
                │           │
                │           └─ imagetaker.py
                │           │  
                │           └─ simulation_classifier.ipynb         
                │           │             
                │           │    
                │           │─ SimulationClassifierTest.ipynb  
                │
                └────good copy tracker.pdf

   
```

### Image Classification using MobileNetv2

We have used im
ML was implemented in this project to classify simulations, by labelling them vision.py will render the simulation by detecting the arucoMarkers.


### Computer Vision


```
./computer-vision/vision.py 
```
 
#### ArUco Tracker

AugmentEd uses [ArUco Markers](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html) to identify the correspondence between the real environment coordinates and projection of science simulations (or experiments). For this project arUco markers were generated via `dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)` 

![](./computer-vision/img/aruco_markers.png)

The markers are detected in `vision.py`:
```
33      print("[INFO] initializing marker detector...")
34      arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
35      arucoParams = cv2.aruco.DetectorParameters_create()
36      print("[INFO] accessing video stream...")
```

and depending on which simulation is identified by MobileNet an animation is rendered on the piece of paper. 
To achive this we made use of projective geometry, in particular homography. A geametrical transformation that preserve the 

#### Homography

Homography is an isomorphism, i.e. a transformation of projective space, that allows us to project from a surface to the other by preserving its map. To achive this we have use


