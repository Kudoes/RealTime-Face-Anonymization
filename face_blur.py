import cv2
import sys
from dnn_face_detector import DNN_Face_Detector as dnn_detector


# Set the required tolerance (confidence cut-off to display a face)
TOLERANCE = 0.45

# Select what to do with detected faces
# 'd' =  draw rectangle; 'b' = Gaussian blur the face; 'p' = pixellate the face
DETECTION = "p"

# Set to True if input is video
# Set to False if input is just a single picture
SRC = True

# Set to 0 if using webcam/live video, or provide input AVI file if using a video
# Also set the filename of the saved output video
VIDEO_INPUT = 0  # "resources/videos/shortclip.mp4"  # 0
OUTPUT_FILE_NAME = "webcam.avi"

# If the input is just a single picture, set parameters here
SRC_PIC = "resources/test_faces_blank.jpg"
RES_NAME = "misc/test_faces_pixellated.jpg"


def main():
    global SRC

    print("===== Starting Network Configuration =====")

    # Get the video input (either webcam or video file)
    input_src, output_file = setVideoInput()
    length = int(input_src.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the dnn_detector in the dnn_face_detector.py class with parameters defined above
    detector = dnn_detector(tolerance=TOLERANCE, handle_detection=DETECTION)

    print("===== Starting Face Detection =====")

    frame_number = 0
    while SRC:

        # Read next frame
        ret, frame = input_src.read()
        frame_number += 1

        # Quit when the input video file ends (if not using webcam)
        if VIDEO_INPUT != 0:
            if not ret:
                break

        # Convert frame to RGB from BGR (necessary for pre-processing for the network)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        detector.detect_faces(frame)

        # Write to console the current frame being processed
        print("Writing frame {} / {}".format(frame_number, length))

        # Convert the frame back to BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the output file and also display in monitoring window
        output_file.write(rgb_frame)
        cv2.imshow("Output Video (Press 'q' to exit program)", rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            SRC = False

    else:
        frame = cv2.imread(SRC_PIC)
        frame_number += 1

        # Convert frame to RGB from BGR (necessary for pre-processing for the network)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        detector.detect_faces(frame)

        # Write to console the current frame being processed
        print("Writing frame {} / {}".format(frame_number, length))

        # Convert the frame back to BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the output file and also display in monitoring window
        cv2.imwrite(RES_NAME, rgb_frame)
        # cv2.imshow("Output Video (Press 'q' to exit program)", rgb_frame)

    input_src.release()
    output_file.release()
    cv2.destroyAllWindows()

    print("===== Program Closing =====")


def setVideoInput():
    try:
        if VIDEO_INPUT == 0:
            # Webcam input
            input_src = cv2.VideoCapture(0)
            length = int(input_src.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create an output movie file
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_movie = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, 30, (640, 480))

        else:  # Input AVI video
            input_src = cv2.VideoCapture(VIDEO_INPUT)
            length = int(input_src.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(input_src.get(cv2.CAP_PROP_FPS))
            w = int(input_src.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(input_src.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create an output movie file
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_movie = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, 30, (w, h))
    except Exception as e:
        print(e)
        print("Error encountered. Please inspect video file and try again.")
        sys.exit(1)

    return input_src, output_movie


if __name__ == "__main__":
    main()
