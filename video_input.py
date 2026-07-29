import cv2
import logging

class VideoInput:
    """
    Handles video input from file or webcam.
    """
    def __init__(self, source=0):
        """
        Initialize video capture.

        Args:
            source (int or str): 0 for webcam, or path to video file.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logging.warning(f"Failed to open {source}, trying webcam")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("Failed to open video source")
                raise ValueError("Video source not available")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Video input initialized with FPS: {self.fps}, size: {self.width}x{self.height}")

    def read_frame(self):
        """
        Read the next frame.

        Returns:
            tuple: (ret, frame) where ret is bool, frame is numpy array.
        """
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to read frame")
        return ret, frame

    def release(self):
        """
        Release the video capture.
        """
        self.cap.release()
        logging.info("Video input released")