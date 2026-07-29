from ultralytics import YOLO
import torch
import logging
import cv2
from preprocessing import ImagePreprocessor

class HumanVerifier:
    """
    Simple human verification based on bounding box properties.
    """
    def __init__(self, min_aspect_ratio=0.1, max_aspect_ratio=2.0, min_area_ratio=0.0001):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area_ratio = min_area_ratio
        logging.info("Human verifier initialized with aspect ratio filtering")

    def is_human_like(self, bbox, frame_shape):
        """
        Check if bbox looks like a human based on aspect ratio and size.

        Args:
            bbox (list): [x1, y1, x2, y2]
            frame_shape (tuple): (height, width)

        Returns:
            bool: True if human-like
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        
        frame_area = frame_shape[0] * frame_shape[1]
        bbox_area = width * height
        area_ratio = bbox_area / frame_area
        if area_ratio < self.min_area_ratio:
            return False
        
        return True

class ObjectDetector:
    """
    Object detection using YOLO for person class with human verification.
    """
    def __init__(self, model_path='yolov8n.pt', conf=0.5, use_preprocessing=True, use_mock_detections=False, use_human_verification=True):
        """
        Initialize YOLO model with preprocessing and human verification.

        Args:
            model_path (str): Path to YOLO model.
            conf (float): Confidence threshold (increased for better accuracy).
            use_preprocessing (bool): Enable preprocessing for robustness.
            use_mock_detections (bool): Return fake detections when inference fails.
            use_human_verification (bool): Use bbox properties to verify humans.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf = conf
        self.use_preprocessing = use_preprocessing
        self.use_mock_detections = use_mock_detections
        self.use_human_verification = use_human_verification
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logging.warning(f"Failed to load YOLO model: {e}. Using mock detector.")
            self.model = None
        
        if self.use_preprocessing:
            self.preprocessor = ImagePreprocessor()
        
        if self.use_human_verification:
            self.verifier = HumanVerifier()
        else:
            self.verifier = None
        
        logging.info(f"Object detector initialized on {self.device} with preprocessing: {use_preprocessing}, human verification: {use_human_verification}")

    def detect(self, frame):
        """
        Detect persons in the frame with human verification.

        Args:
            frame (numpy array): Input frame.

        Returns:
            list: List of detections [(bbox, conf), ...]
        """
        # Apply preprocessing for robustness
        if self.use_preprocessing:
            frame = self.preprocessor.preprocess(frame)
        
        logging.debug(f"Detector called with frame shape {frame.shape}, model is None: {self.model is None}, use_mock_detections: {self.use_mock_detections}")
        
        using_mock = False
        if self.model is None or self.use_mock_detections:
            if self.use_mock_detections:
                h, w = frame.shape[:2]
                # Create mock humans at different distances for testing 5-tier risk system
                # These correspond to the humans drawn in run_demo.py
                mock_humans = [
                    # CRITICAL: Very close (< 5m) AND center laterally (< 3m)
                    ([315, 475, 325, 480], 0.95),  # Extreme bottom-center
                    # HIGH_WARNING: Close (5-15m) AND within lateral zone (< 8m)
                    ([300, 400, 340, 440], 0.90),  # Lower center area
                    # WARNING: Medium (15-25m) - middle
                    ([350, 200, 385, 240], 0.85),  # Middle-right
                    # LOW_WARNING: Far (25-40m) - upper middle
                    ([260, 110, 295, 150], 0.80),  # Upper-left
                    # SAFE: Very far (>40m) - very top
                    ([450, 20, 475, 45], 0.75),  # Top-right
                ]
                detections = mock_humans
                logging.warning("Using mock detections for testing 5-tier risk system.")
                using_mock = True
            else:
                detections = []
                logging.warning("YOLO model unavailable; returning no detections.")
        else:
            try:
                half = self.device == 'cuda'
                results = self.model(frame, classes=[0], conf=self.conf, device=self.device, half=half, verbose=False)  # person class
                detections = []
                for result in results:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append((bbox, conf))
                logging.info(f"YOLO detected {len(detections)} persons with conf >= {self.conf}")
                if len(detections) == 0:
                    logging.info("No persons detected by YOLO - check camera view or confidence threshold")
            except Exception as e:
                logging.warning(f"YOLO detection failed: {e}. Using fallback.")
                detections = []
                logging.warning("YOLO detection failed; returning no detections.")
        
        # Human verification to filter detections (skip for mock)
        if self.verifier and detections and not using_mock:
            verified_detections = []
            for bbox, conf in detections:
                if self.verifier.is_human_like(bbox, frame.shape):
                    verified_detections.append((bbox, conf))
                else:
                    logging.debug(f"Filtered out non-human-like detection at {bbox}")
            detections = verified_detections
        
        logging.debug(f"Detected {len(detections)} verified persons")
        return detections

    def _filter_detections(self, detections, frame_shape):
        """
        Filter detections based on realistic human size and aspect ratio.
        
        Args:
            detections (list): [(bbox, conf), ...]
            frame_shape (tuple): (height, width)
            
        Returns:
            list: Filtered detections
        """
        h, w = frame_shape[:2]
        filtered = []
        
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Size check: too small or too large
            area = width * height
            min_area = (w * h) * 0.001  # 0.1% of frame
            max_area = (w * h) * 0.5    # 50% of frame
            if area < min_area or area > max_area:
                continue
            
            # Aspect ratio check: realistic for humans (standing/walking)
            aspect_ratio = width / height if height > 0 else 0
            if not (0.2 <= aspect_ratio <= 1.5):  # Allow some variation
                continue
            
            # Position check: not too high (sky) or too low (ground)
            center_y = (y1 + y2) / 2
            if center_y < h * 0.1 or center_y > h * 0.9:  # top/bottom 10%
                continue
            
            filtered.append((bbox, conf))
        
        return filtered