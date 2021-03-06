# Real-Time Face Detection and Blurring
This program utilizes OpenCV's DNN to detect faces in (1) photos, (2) video files and/or (3) real-time video input and then anonymize them by either Gaussian blurring or pixelation.

Final Project for CS 445 @ UIUC.

# Repository Contents

The contents of this repository are as follows:

1. **misc/**: Miscellaneous files for GitHub display purposes.
2. **model_config/**: The configuration files for the DNN (such as the weights file).
3. **resources/**: Folder to place any resources used as input.
4. **dnn_face_detector.py**: The DNN class which is utilized to detect faces in a given frame.
5. **face_blur.py**: The main program that will be run to do the detection and anonymization. Parameters/input can be tuned here.
6. **download_pb_file.py**: A program that will download the model weights file needed to initialize the neural network.
7. **requirements.txt**: A requirements file that will download all the dependencies.

# Installation

## Requirements:

1. This program was tested on both Windows 10 and MacOS. It has not been tested on Linux, but that is not to say that it will not work on Linux operating systems.

2. Python 3.7

## Installation Procedure:

1. Clone this repo

2. Run `pip install -r requirements.txt`

    a. Ideally, you should do this in a fresh virtual environment.

# Run the Program

To run the program, simply navigate into `face_blur.py` and adjust the parameters defined at the top of the file, and run `python face_blur.py`. The parameters:

1. **TOLERANCE**: The level of tolerance at which faces will be detected (between 0 and 1). 

    a. Lower values will detect faces more frequently (i.e. at different angles/sizes), but also detect potential false positives more frequently. 
    
    b. Higher values will detect faces at different angles/sizes less frequently, but they will be more accurate detections.

2. **DETECTION**: The method of post-detection anonymization.

    a. 'd': Draw a rectangle around the detected face with the confidence displayed above. This **does not** anonymize.
    
    b. 'b': Gaussian-blur the face.
    
    c. 'p': Pixelate the face.

3. **SRC_IS_VID**: A flag to indicate whether the input source is a video.

    a. Set to **true** if the input source is a video (either webcam or video file)
    
    b. Set to **false** if the input is a picture.

4. **INPUT_NAME**: Specify the input video (stream/file) **or** the input picture.

    a. Set to '0' if using a webcam.
    
    b. Set to the location of the input video file if using a video file (E.g. "resources/video.mp4").
    
    c. Set to the location of the input picture if using a picture (E.g. "resources/people.jpg").

5. **OUTPUT_FILE_NAME**: The name of the output file.

    a. When output is a video, specify the output file as a video file with .avi extension.
    
    b. When output is a picture, specify the output file with a graphical extension (.jpg, .png, etc.).

# Example Results: Pictures
A stock photo with five people was one of the tests used to determine the accuracy of the face detection done by the DNN. The results of (1) detection, (2) blurring, and (3) pixelization are shown below.

## Original Picture

<p align="center">
<img src="https://github.com/Kudoes/RealTime-Face-Blurring/blob/master/misc/test_faces_blank.jpg" width="500" class="center">
</p>

## Face Detection

<p align="center">
<img src="https://github.com/Kudoes/RealTime-Face-Blurring/blob/master/misc/test_faces_detected.jpg" width="500">
</p>

## Faces Blurred

<p align="center">
<img src="https://github.com/Kudoes/RealTime-Face-Blurring/blob/master/misc/test_faces_blurred.jpg" width="500">
</p>

## Faces Pixelated

<p align="center">
<img src="https://github.com/Kudoes/RealTime-Face-Blurring/blob/master/misc/test_faces_pixellated.jpg" width="500">
</p>

# Video Results:

Youtube Link: [Real-time Face Detection and Anonymization Results](https://www.youtube.com/watch?v=bKoHkIEWQRk)



----------------------------------------------------------------------------------

Resources:
1. https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
2. https://github.com/opencv/opencv/tree/master/samples/dnn
3. https://caffe2.ai/docs/tutorial-models-and-datasets.html
4. https://kezunlin.me/post/9054e84f/
5. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
6. https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

The videos utilized in the results:

1. [What Makes Carmelo Anthony So Good? - New York Knicks](https://www.youtube.com/watch?v=aSmNgxOZfTs)

2. [Kawhi Leonard, Kyle Lowry talk Raptors’ rollercoaster season - ESPN](https://www.youtube.com/watch?v=sm2e4NT1yho&t=51s)
