# 'main.py' -> Camera pose Estimation
```main.py``` contains functions to estimate the camera pose from the chessboard video. When executed, the script renders a 3D rectangular prism (cuboid) on top of the chessboard, allowing you to visualize and verify the camera's orientation and position in real-time.

I have intentionally separated this logic from gojo_dance.py. Because running pose estimation and the primary AR animation simultaneously can lead to a cluttered visual output.

# Gojo-is-dancing
Gojo is dancing with openCV

## 1. Library Import
```
import cv2 as cv
import numpy as np
import imageio
import os
import shutil
```
This time, I use '''imageio''' and '''os''' and '''shutil'''.
- ```imageio``` is used to get the png(frames) from gif.
- ```os``` and ```shutil``` is used to manage OS and file system.

## 2. Significant Variables
```criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)```

```criteria``` is used to set the criteria for chessboard corner refinement.

```
K = np.array([ 
    [892.36877456, 0.0, 961.8528907],
    [0.0, 894.46553008, 541.85538616],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([-0.05734935, 0.23092948, 0.00236224, 0.00213651, -0.38708988])
```
```K``` and ```dist_coeffs``` is already derived in last file. If you did not derive this variables, you have to put the function to get ```K``` and ```dist_coeffs```. But, this function is taken so long time to complie in python, so I use values in last file.

```alpha = 0.6```
```alpha```filter is used to ease ```rvec``` and ```tvec```. ```rvec``` is rotation vector that use 'Rodrigues' formula and ```tvec``` is translation vector to get the coordinate that how far from chessboard to camera

```fast_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE```
You can read about ```fast_flags``` ```3. Functions``` section.


## 3. Functions
``` cv.CALIB_CB_... ```
1. ```cv.CALIB_CB_ADAPTIVE_THRESH```

This function is used when ambient lighting is uneven. I included it because the lighting in my recording environment was inconsistent.

2. ```cv.CALIB_CB_FAST_CHECK```

This function is the key to improving processing speed; it quickly skips frames if no chessboard is detected. (Though, in this specific video, the chessboard is detected at all times.)

3. ```cv.CALIB_CB_NORMALIZE_IMAGE```

This function adjusts the image contrast. It normalizes overly bright or dark images to ensure smooth and accurate detection.

```def gif_to_png_sequence(gif_path, output_dir) ```
This function is used to get the images from GIF. The GIF is decomposed into individual frames and saved as PNG files in the './pngs' directory. 

```def load_png_sequence(folder):```
This function is used to load the list of images

```def remove_png_files(folder_path):```
This function is used to remove the png files in ```./pngs``` directory.

## 4. Used Image  
I use the GIF files that is 'Gojo Satoru' from 'Jujutsu Kaisen' dancing.
<img src="C:\Users\peter\Desktop\과기대\3-1\컴퓨터비전\과제\frames_screenshot.png">

## 5. Limitation in my file
Intermittent Model Flickering (1-Frame Disappearance):
You may notice the Gojo Satoru AR model occasionally flickering or disappearing for a single frame. This is primarily due to:

Corner Detection Instability(체스보드 검출 오류): In specific frames, environmental factors such as motion blur or light reflection may cause cv2.findChessboardCorners() to fail, resulting in a temporary loss of the transformation matrix.

Pose Estimation Sensitivity(자세 민감도): The AR rendering is highly dependent on the continuous detection of the calibration pattern. Any occlusion or extreme camera angles can disrupt the tracking pipeline for a fraction of a second.

### The first video of chessboard
is uploaded as 'chessboard1.mp4'
### Result video 
is uploaded as 'output_high_res.mp4'
