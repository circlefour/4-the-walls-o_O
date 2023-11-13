"""
configuration settings for the OpenCV DNN face detection model
"""

import cv2 as cv

class FaceDetectionConfig:
    def __init__(
            self,
            config,
            model_path='face_detection_yunet_2022mar.onnx',
            config_file='',
            input_size=(320,320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000):
        
        self.scale = config.scale
        self.model_path = model_path
        self.config_file = config_file
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

class FaceDetection:
    def __init__(self, detection_config):
        self.detector = cv.FaceDetectorYN.create(
            detection_config.model_path,
            detection_config.config_file,
            detection_config.input_size,
            detection_config.score_threshold,
            detection_config.nms_threshold,
            detection_config.top_k
        )
        self.scale = detection_config.scale

    def detect_faces(self, frame):
        frame_width = int(frame.shape[1] * self.scale)
        frame_height = int(frame.shape[0] * self.scale)
        self.detector.setInputSize([frame_width, frame_height])
        return self.detector.detect(frame)