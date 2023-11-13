import cv2 as cv
import numpy as np
import imutils

# misc functions
from glitchers import *
from img_processing import *
from frame_modifiers import *

# importing modules
from display.display_config import DisplayConfig
from display.display_window import DisplayWindow
from face_detection.face_detection_config import FaceDetection, FaceDetectionConfig
from video_streams import VideoStreams

def main():
    config = DisplayConfig()
    display_window = DisplayWindow(config, 'experiments')
    display_window.create_window()
    detection_config = FaceDetectionConfig(config)
    face_detection = FaceDetection(detection_config)

    # FIXME: add argparser module to grab input on the fly : perhaps
    sources = {'webcam': 0}

    glitch_functs = [
        (face_cage, {}),
        # iterations > 100 is pretty much what i was running
        (block_glitch, {'iterations': 5}),
    ]

    filter_functs = [
        (linear_combination_of_frames, {}),
    ]

    with VideoStreams(sources) as caps:
        on_start()

        # esc key
        while cv.waitKey(1) != 27:

            frames = {source: cap.read() for source, cap in caps.items()}

            for source, frame in frames.items():
                frame = imutils.resize(frame, width=int(frame.shape[1] * config.scale), height=int(frame.shape[0] * config.scale))

                # Inference
                faces = face_detection.detect_faces(frame)

                # Draw results on the input
                apply_face_glitch(frame, faces, glitch_functs)
                frame[0:frame.shape[0],0:frame.shape[1]//2] = apply_frame_filter(frame[0:frame.shape[0],0:frame.shape[1]//2], filter_functs)
                
                display_window.display_frame(frame)
                
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()