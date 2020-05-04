## Utilizes/modifies code and configuration files found on official OpenCV GitHub Repository
## https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector


import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import download_pb_file as download_config

# DNN Face Detector Class
class DNN_Face_Detector:
    def __init__(self, tolerance=0.5, handle_detection="d"):
        # The deep network
        self.NET = None

        # Percent tolerance (i.e., 0.5 = consider all results > 50% conf as faces)
        self.TOLERANCE = tolerance

        # Either 'd' for 'detect/rectangle' or 'b' for blur
        self.HANDLE_DETECTION = handle_detection

        # Initialize the model with pre-built weights
        self.init_model()

    # Initialize (download if not already present) the configuration files for the DNN
    # Also, ensure model parameters are valid
    def init_model(self):

        # Now ensure supplied parameters to this class are valid
        try:
            # Check if configuration files exist; if not, download them
            res = download_config.main()

            # If successfully downloaded initialization file, load pre-trained model
            if res:
                PB_FILE = "model_config/opencv_face_detector_uint8.pb"
                PB_TEXT_FILE = "model_config/opencv_face_detector.pbtxt"
                self.NET = cv2.dnn.readNetFromTensorflow(PB_FILE, PB_TEXT_FILE)
            else:
                print("Unable to download initialization file. Exiting...")
                sys.exit(1)

            if not 0 <= self.TOLERANCE <= 1:
                print(
                    "Incorrect TOLERANCE. Please ensure tolerance is a float between 0 and 1."
                )
                sys.exit(1)

            if self.HANDLE_DETECTION not in ["d", "b", "p"]:
                print(
                    "Incorrect HANDLE_DETECTION. Please ensure HANDLE_DETECTION is one of: 'd' (detect), 'b' (blur) or 'p' (pixellate)."
                )
                sys.exit(1)
        except Exception as e:
            print("Invalid parameters. Please provide valid inputs.")
            print(e)
            sys.exit(1)

    # Detect faces in a frame and draw a box around it
    def detect_faces(self, frame):

        # Get height/width of the frame to resize the result back to original dimensions
        h, w = frame.shape[0], frame.shape[1]

        # Use the recommended frame settings for this pre-trained model to generate a blob
        blob = cv2.dnn.blobFromImage(
            image=cv2.resize(frame, (300, 300)),  # Resize the frame to 300x300
            scalefactor=1.0,  # 1.0 = no scaling
            size=(300, 300),  # Size of image that network expects
            mean=(
                123.0,
                177.0,
                104.0,
            ),  # what is subtracted from each channel (RGB order)
        )

        # Set frame blob as input to the network
        self.NET.setInput(blob)

        # Pass the blob through the network and get results (results = 4D matrix of which we only need last 2)
        detections = self.NET.forward()

        # Format of detections:
        # detections = [_, _, potential_detections, [_, _, confidence, x1, y1, x2, y2]]
        num_detections = detections.shape[2]

        # Iterate over all faces detected in the frame
        for i in range(0, num_detections):

            # For each potential detection, get the confidence of it being a face
            confidence = detections[0, 0, i, 2]

            # If the confidence is > 0.5, it means we can be confident it has a face
            if confidence > self.TOLERANCE:

                # Extract co-ordinates of the bounding box and scale back to original resolution
                x1, y1, x2, y2 = (
                    int(round(detections[0, 0, i, 3] * w)),
                    int(round(detections[0, 0, i, 4] * h)),
                    int(round(detections[0, 0, i, 5] * w)),
                    int(round(detections[0, 0, i, 6] * h)),
                )

                # Now either draw a rectangle or blur the face depending on toggle
                if self.HANDLE_DETECTION == "d":
                    draw_rectangle(frame, confidence, x1, y1, x2, y2)
                elif self.HANDLE_DETECTION == "b":
                    blur_face(frame, x1, y1, x2, y2)
                elif self.HANDLE_DETECTION == "p":
                    pixellate_face(frame, x1, y1, x2, y2)

        # Return modified frame with detected faces
        return frame


# Gaussian blur the detected face
def blur_face(frame, x1, y1, x2, y2):
    w, h = frame[y1:y2, x1:x2].shape[:2]
    w2, h2 = 10, 10

    # Blur the region of the image that contains the face
    face_image = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)

    # Put the blurred face region back into the frame image
    frame[y1:y2, x1:x2] = face_image

    return frame


# Pixellate the detected face by shrinking the image and then re-sizing it back to original dimensions
def pixellate_face(frame, x1, y1, x2, y2):
    w, h = frame[y1:y2, x1:x2].shape[:2]
    w2, h2 = 10, 10

    # Shrink the image to above dimensions
    temp = cv2.resize(frame[y1:y2, x1:x2], (h2, w2), interpolation=cv2.INTER_LINEAR)

    # Re-size the image with nearest-neighbor interpolation
    face_image = cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST)

    # Put the pixellated face region back into the frame image
    frame[y1:y2, x1:x2] = face_image

    return frame


# Draw a rectangle around the face detection and display confidence
# Utilization explanation found on: https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
def draw_rectangle(frame, confidence, x1, y1, x2, y2):

    # Draw rectangle around the face co-ordinates
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    cv2.putText(
        frame,  # Put the text on the given frame
        "conf: {}%".format(round(confidence * 100), 1),
        (x1, y1 - 10),  # Display confidence above box
        cv2.FONT_HERSHEY_SIMPLEX,  # Font is this
        0.45,  # Scale of font
        (255, 0, 0),  # Color = red
        1,  # Line thickness of 1
    )

    return frame
