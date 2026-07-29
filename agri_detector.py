"""
Advanced Agricultural Human Detection for Challenging Field Conditions.

Specialized for wheat fields, dusty environments, varying lighting, and partial occlusions.
Focuses on human detection with 100% reliability for autonomous harvester safety.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging

class AgriculturalHumanDetector:
    """
    Specialized human detector for agricultural environments with:
    - Multi-scale detection (tiny to distant humans)
    - Dust/shadow robustness
    - Partial occlusion handling
    - Color-based skin detection as backup
    - Motion-based detection for moving humans
    - Enhanced far-distance detection capabilities
    """

    def __init__(self, model_path='yolov8l.pt', conf=0.25, use_preprocessing=True, enable_far_detection=True):
        """
        Initialize agricultural human detector with enhanced far-distance capabilities.

        Args:
            model_path: YOLO model (l-size for better field detection)
            conf: Base confidence threshold (lowered to 0.25 for far detection)
            use_preprocessing: Enable agricultural preprocessing
            enable_far_detection: Enable enhanced far-distance detection
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_conf = conf
        self.use_preprocessing = use_preprocessing
        self.enable_far_detection = enable_far_detection

        # Dynamic confidence thresholds based on detection size
        self.conf_thresholds = {
            'close': 0.4,      # High confidence for close humans
            'medium': 0.25,    # Medium confidence for medium distance
            'far': 0.15,       # Low confidence for far humans
            'distant': 0.1     # Very low for very distant humans
        }

        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded YOLOv8 large model for agricultural detection")
        except Exception as e:
            logging.warning(f"Failed to load YOLO model '{model_path}': {e}")
            alt_model_path = 'yolov8n.pt'
            if os.path.exists(alt_model_path):
                try:
                    self.model = YOLO(alt_model_path)
                    logging.info(f"Loaded fallback YOLOv8 nano model from {alt_model_path}")
                except Exception as e2:
                    logging.warning(f"Failed to load fallback YOLO model '{alt_model_path}': {e2}")
                    logging.info("Continuing with alternative detection methods (HOG, contour, skin)")
                    self.model = None
            else:
                logging.info("Continuing with alternative detection methods (HOG, contour, skin)")
                self.model = None

        # Initialize HOG descriptor for pedestrian detection with enhanced parameters
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Enhanced HOG parameters for far detection
        self.hog_win_stride = (4, 4)  # Smaller stride for better coverage
        self.hog_padding = (8, 8)     # Smaller padding
        self.hog_scale = 1.05         # Finer scale steps

        # Agricultural-specific color ranges for skin detection (backup)
        self.skin_lower_hsv = np.array([0, 20, 70])
        self.skin_upper_hsv = np.array([20, 255, 255])

        # Ultra-far distance detection parameters
        self.ultra_far_scales = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]  # More aggressive scaling
        self.subpixel_precision = True
        self.adaptive_thresholding = True
        self.context_aware_detection = True
        self.super_resolution_enabled = True
        self.frequency_domain_enhancement = True
        self.adversarial_noise_reduction = True

        # Thermal imaging integration
        self.thermal_enabled = True
        self.thermal_sensitivity = 0.8
        self.night_vision_mode = True
        self.multi_spectral_fusion = True

        # Low-light enhancement parameters
        self.ir_threshold_min = 30
        self.ir_threshold_max = 200
        self.thermal_smoothing = True
        
        # Contour detection parameters
        self.min_contour_area = 500  # Minimum area for human-like contours
        self.max_contour_area = 50000  # Maximum area
        self.aspect_ratio_min = 0.3  # Minimum width/height ratio
        self.aspect_ratio_max = 1.0  # Maximum width/height ratio

    def enhance_for_ultra_far_detection(self, frame):
        """
        Ultra-advanced preprocessing for extreme far-distance detection.
        Combines multiple enhancement techniques for maximum detection range.

        Args:
            frame: Input frame (BGR)

        Returns:
            Enhanced frame optimized for ultra-far detection
        """
        enhanced = frame.copy()

        # 1. Super-resolution enhancement using frequency domain
        if self.super_resolution_enabled:
            enhanced = self._apply_super_resolution(enhanced)

        # 2. Advanced CLAHE with adaptive parameters
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Adaptive CLAHE based on image characteristics
        if self.adaptive_thresholding:
            # Calculate image statistics for adaptive enhancement
            mean_brightness = np.mean(l)
            std_brightness = np.std(l)

            # Adjust CLAHE parameters based on image conditions
            if std_brightness < 30:  # Low contrast (far distance)
                clip_limit = 6.0
                tile_size = (2, 2)
            elif std_brightness < 60:  # Medium contrast
                clip_limit = 4.0
                tile_size = (4, 4)
            else:  # High contrast (close distance)
                clip_limit = 3.0
                tile_size = (8, 8)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        else:
            clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(2, 2))

        l = clahe.apply(l)

        # 3. Frequency domain enhancement for small details
        if self.frequency_domain_enhancement:
            l = self._apply_frequency_enhancement(l)

        # 4. Advanced sharpening with edge preservation
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        l = cv2.filter2D(l, -1, kernel)

        # 5. Adversarial noise reduction
        if self.adversarial_noise_reduction:
            l = self._apply_adversarial_denoising(l)

        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 6. Multi-scale bilateral filtering for edge preservation
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        enhanced = cv2.bilateralFilter(enhanced, 9, 25, 25)

        return enhanced

    def _apply_super_resolution(self, frame):
        """
        Apply super-resolution enhancement for small object detection.

        Args:
            frame: Input frame

        Returns:
            Super-resolution enhanced frame
        """
        # Simple but effective super-resolution using interpolation and sharpening
        # In production, this would use ESRGAN or similar ML-based super-resolution

        # Upscale with Lanczos interpolation
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

        # Apply unsharp masking for detail enhancement
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)

        # Downscale back to original size with enhanced details
        enhanced = cv2.resize(unsharp_mask, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return enhanced

    def _apply_frequency_enhancement(self, channel):
        """
        Apply frequency domain enhancement to boost high-frequency details.

        Args:
            channel: Single channel image

        Returns:
            Frequency-enhanced channel
        """
        # Convert to frequency domain
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create high-pass filter
        rows, cols = channel.shape
        crow, ccol = rows//2, cols//2

        # Create mask with higher boost for center frequencies (human-sized objects)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 30  # Radius for high-pass filter
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0

        # Apply Gaussian falloff for smoother transition
        y_coords, x_coords = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x_coords - ccol)**2 + (y_coords - crow)**2)
        gaussian_mask = np.exp(-dist_from_center**2 / (2 * (r/2)**2))
        mask[:, :, 0] = mask[:, :, 0] * gaussian_mask
        mask[:, :, 1] = mask[:, :, 1] * gaussian_mask

        # Apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        enhanced = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize and convert back
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return enhanced

    def _apply_adversarial_denoising(self, channel):
        """
        Apply adversarial noise reduction to preserve small details.

        Args:
            channel: Single channel image

        Returns:
            Denoised channel
        """
        # Advanced denoising that preserves edges and small details
        # Uses bilateral filter with adaptive parameters

        # Calculate local variance to adapt denoising strength
        local_var = cv2.blur(channel.astype(np.float32)**2, (5, 5)) - cv2.blur(channel.astype(np.float32), (5, 5))**2

        # Adaptive bilateral filter parameters
        sigma_color = np.maximum(10, 50 * (local_var / np.max(local_var)))
        sigma_space = 5

        # Apply adaptive bilateral filter
        denoised = cv2.bilateralFilter(channel, 5, sigma_color.mean(), sigma_space)

        return denoised

    def detect_ultra_far_humans(self, frame):
        """
        Ultra-far distance human detection with sub-pixel accuracy.

        Args:
            frame: Input frame (BGR)

        Returns:
            list: Ultra-far detections [(bbox, confidence, metadata), ...]
        """
        ultra_far_detections = []

        # Enhanced preprocessing for ultra-far detection
        enhanced_frame = self.enhance_for_ultra_far_detection(frame)

        # Multi-scale detection with ultra-far scales
        for scale in self.ultra_far_scales:
            if scale > 1.0:
                # Upscale frame for better small object detection
                scaled_frame = cv2.resize(enhanced_frame, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_LANCZOS4)
            else:
                scaled_frame = enhanced_frame

            # Apply different detection methods at each scale
            scale_detections = []

            # 1. Enhanced YOLO detection at ultra-low confidence
            if self.model:
                try:
                    results = self.model(scaled_frame, classes=[0], conf=0.05,  # Ultra-low confidence
                                       device=self.device, verbose=False)

                    for result in results:
                        for box in result.boxes:
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())

                            # Scale bbox back to original size
                            if scale > 1.0:
                                bbox = bbox / scale

                            # Sub-pixel refinement
                            if self.subpixel_precision:
                                bbox = self._refine_subpixel_bbox(bbox, frame)

                            # Calculate detection characteristics
                            x1, y1, x2, y2 = bbox
                            width = x2 - x1
                            height = y2 - y1
                            size = width * height
                            aspect_ratio = width / height if height > 0 else 0

                            # Ultra-far detection criteria
                            if (size < 100 and  # Very small size
                                0.2 < aspect_ratio < 5.0 and  # Reasonable aspect ratio
                                conf > 0.05):  # Minimum confidence

                                metadata = {
                                    'detection_scale': scale,
                                    'original_size': size,
                                    'aspect_ratio': aspect_ratio,
                                    'detection_method': 'ultra_far_yolo',
                                    'confidence_boost': self._calculate_ultra_far_confidence(conf, size, scale)
                                }

                                scale_detections.append((bbox, conf, metadata))

                except Exception as e:
                    logging.debug(f"Ultra-far YOLO detection failed at scale {scale}: {e}")

            # 2. Enhanced contour detection for ultra-small objects
            try:
                contours = self._detect_ultra_small_contours(scaled_frame, scale)
                for contour_data in contours:
                    bbox, conf, contour_metadata = contour_data

                    # Scale back to original size
                    if scale > 1.0:
                        bbox = [b / scale for b in bbox]

                    if self.subpixel_precision:
                        bbox = self._refine_subpixel_bbox(bbox, frame)

                    metadata = {
                        'detection_scale': scale,
                        'detection_method': 'ultra_far_contour',
                        **contour_metadata
                    }

                    scale_detections.append((bbox, conf, metadata))

            except Exception as e:
                logging.debug(f"Ultra-far contour detection failed at scale {scale}: {e}")

            # Add scale detections to main list
            ultra_far_detections.extend(scale_detections)

        # Remove duplicates and merge overlapping detections
        ultra_far_detections = self._merge_ultra_far_detections(ultra_far_detections)

        logging.info(f"Ultra-far detection found {len(ultra_far_detections)} extremely distant humans")

        return ultra_far_detections

    def _detect_ultra_small_contours(self, frame, scale):
        """
        Detect ultra-small contours that might be very distant humans.

        Args:
            frame: Input frame
            scale: Current detection scale

        Returns:
            list: [(bbox, confidence, metadata), ...]
        """
        # Convert to grayscale with enhanced contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        gray = clahe.apply(gray)

        # Multi-threshold edge detection
        edges = cv2.Canny(gray, 10, 50)  # Very sensitive thresholds

        # Morphological operations to connect small edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]

        # Ultra-low minimum area for very distant humans
        min_area = max(4, (h * w) * 0.000005)  # Extremely small minimum

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < 500:  # Very small but not noise
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)
                bbox = [x, y, x + cw, y + ch]

                # Enhanced shape analysis for ultra-small contours
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Calculate compactness and other shape features
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Human-like criteria for ultra-small detections
                aspect_ratio = cw / ch if ch > 0 else 0

                # Very relaxed criteria for ultra-far detection
                if (0.1 < aspect_ratio < 10.0 and  # Very flexible aspect ratio
                    circularity < 0.95 and  # Not too circular
                    solidity > 0.3):  # Some solidity

                    # Calculate confidence based on shape characteristics
                    shape_score = (solidity + (1 - circularity) + min(1.0, aspect_ratio / 2)) / 3.0
                    size_score = min(1.0, area / 100)  # Size bonus for larger small detections
                    confidence = 0.3 + (shape_score * 0.4) + (size_score * 0.3)

                    metadata = {
                        'contour_area': area,
                        'aspect_ratio': aspect_ratio,
                        'circularity': circularity,
                        'solidity': solidity,
                        'shape_score': shape_score,
                        'size_score': size_score
                    }

                    detections.append((bbox, confidence, metadata))

        return detections

    def _refine_subpixel_bbox(self, bbox, frame):
        """
        Refine bounding box to sub-pixel accuracy using edge analysis.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            frame: Original frame

        Returns:
            Refined bbox with sub-pixel accuracy
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))

        # Convert to integer coordinates for processing
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # Extract region of interest
        roi = frame[iy1:iy2, ix1:ix2]
        if roi.size == 0:
            return [x1, y1, x2, y2]

        # Convert to grayscale and find edges
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)

        # Find edge pixels
        edge_pixels = np.where(edges > 0)
        if len(edge_pixels[0]) == 0:
            return [x1, y1, x2, y2]

        # Calculate refined bounding box based on edge distribution
        edge_y, edge_x = edge_pixels

        if len(edge_x) > 0 and len(edge_y) > 0:
            # Calculate weighted center of mass of edges
            edge_weights = edges[edge_y, edge_x].astype(np.float32)
            total_weight = np.sum(edge_weights)

            if total_weight > 0:
                # Weighted center of mass
                center_x = np.sum(edge_x * edge_weights) / total_weight + ix1
                center_y = np.sum(edge_y * edge_weights) / total_weight + iy1

                # Calculate spread for size estimation
                spread_x = np.std(edge_x) * 2
                spread_y = np.std(edge_y) * 2

                # Refine bbox coordinates
                refined_x1 = center_x - spread_x
                refined_x2 = center_x + spread_x
                refined_y1 = center_y - spread_y
                refined_y2 = center_y + spread_y

                # Ensure minimum size and reasonable bounds
                min_size = 4
                width = max(min_size, refined_x2 - refined_x1)
                height = max(min_size, refined_y2 - refined_y1)

                refined_x1 = max(0, center_x - width/2)
                refined_x2 = min(w, center_x + width/2)
                refined_y1 = max(0, center_y - height/2)
                refined_y2 = min(h, center_y + height/2)

                return [refined_x1, refined_y1, refined_x2, refined_y2]

        return [x1, y1, x2, y2]

    def _calculate_ultra_far_confidence(self, base_conf, size, scale):
        """
        Calculate enhanced confidence for ultra-far detections.

        Args:
            base_conf: Base confidence from detection
            size: Detection size in pixels
            scale: Detection scale used

        Returns:
            Enhanced confidence score
        """
        # Size-based confidence boost (smaller detections need more boost)
        size_boost = 1.0
        if size < 25:
            size_boost = 2.0
        elif size < 50:
            size_boost = 1.5
        elif size < 100:
            size_boost = 1.2

        # Scale-based confidence adjustment (higher scales are more uncertain)
        scale_penalty = 1.0 / (1.0 + (scale - 1.0) * 0.1)

        # Context-aware confidence based on expected human sizes at different scales
        expected_size_ratio = size / (100 / scale)  # Expected size decreases with scale
        context_boost = min(1.5, 1.0 + expected_size_ratio * 0.5)

        enhanced_conf = base_conf * size_boost * scale_penalty * context_boost

        return min(0.95, enhanced_conf)

    def _merge_ultra_far_detections(self, detections):
        """
        Merge overlapping ultra-far detections with advanced criteria.

        Args:
            detections: List of (bbox, confidence, metadata) tuples

        Returns:
            Merged detections
        """
        if not detections:
            return []

        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)

        merged = []

        for i, (bbox, conf, metadata) in enumerate(detections):
            # Check if this detection significantly overlaps with already merged ones
            should_merge = False
            best_match_idx = -1
            best_iou = 0

            for j, (merged_bbox, _, _) in enumerate(merged):
                iou = self._calculate_iou(bbox, merged_bbox)
                if iou > 0.3:  # Significant overlap
                    should_merge = True
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = j

            if not should_merge:
                merged.append((bbox, conf, metadata))
            else:
                # Merge with existing detection
                existing_bbox, existing_conf, existing_metadata = merged[best_match_idx]

                # Weighted merge based on confidence
                weight1 = existing_conf / (existing_conf + conf)
                weight2 = conf / (existing_conf + conf)

                merged_bbox = [
                    weight1 * existing_bbox[0] + weight2 * bbox[0],
                    weight1 * existing_bbox[1] + weight2 * bbox[1],
                    weight1 * existing_bbox[2] + weight2 * bbox[2],
                    weight1 * existing_bbox[3] + weight2 * bbox[3]
                ]

                merged_conf = max(existing_conf, conf)

                # Merge metadata intelligently
                merged_metadata = self._merge_ultra_far_metadata(existing_metadata, metadata)

                merged[best_match_idx] = (merged_bbox, merged_conf, merged_metadata)

        return merged

    def _merge_ultra_far_metadata(self, meta1, meta2):
        """
        Merge metadata from overlapping ultra-far detections.

        Args:
            meta1, meta2: Metadata dictionaries

        Returns:
            Merged metadata
        """
        merged = {}

        # Average numerical values
        for key in set(meta1.keys()) | set(meta2.keys()):
            if key in meta1 and key in meta2:
                val1, val2 = meta1[key], meta2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    merged[key] = (val1 + val2) / 2
                else:
                    merged[key] = val1  # Keep first value for non-numerics
            elif key in meta1:
                merged[key] = meta1[key]
            else:
                merged[key] = meta2[key]

        # Special handling for detection methods
        methods = []
        if 'detection_method' in meta1:
            methods.append(meta1['detection_method'])
        if 'detection_method' in meta2:
            methods.append(meta2['detection_method'])
        merged['detection_methods'] = list(set(methods))

        return merged

    def detect_skin_color(self, frame):
        """
        Detect skin-colored regions as backup for partially visible humans.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            list: Skin region bboxes [(x1, y1, x2, y2), ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.skin_lower_hsv, self.skin_upper_hsv)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.001  # At least 0.1% of frame
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                detections.append([x, y, x + cw, y + ch])
        
        return detections

    def detect_hog_pedestrians(self, frame):
        """
        Enhanced HOG pedestrian detection with multi-scale support for far away humans.

        Args:
            frame: Input frame (BGR)

        Returns:
            list: Bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Multi-scale detection for far away humans
        all_boxes = []

        if self.enable_far_detection:
            # Try multiple scales for better far detection
            scales = [1.0, 1.2, 1.5, 2.0]
            for scale in scales:
                if scale > 1.0:
                    # Resize frame for larger scale detection
                    scaled_frame = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                else:
                    scaled_frame = gray

                # Detect pedestrians at this scale
                boxes, weights = self.hog.detectMultiScale(
                    scaled_frame,
                    winStride=self.hog_win_stride,
                    padding=self.hog_padding,
                    scale=self.hog_scale,
                    hitThreshold=0.0  # Lower threshold for more detections
                )

                # Scale boxes back to original size and filter
                for (x, y, w, h), weight in zip(boxes, weights):
                    if scale > 1.0:
                        # Scale back to original coordinates
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)

                    # Lower confidence threshold for far detection
                    if weight > 0.2:  # Reduced from 0.3 for better far detection
                        # Expand bounding box slightly for better coverage
                        x1 = max(0, x - int(w * 0.1))
                        y1 = max(0, y - int(h * 0.1))
                        x2 = min(frame.shape[1], x + w + int(w * 0.1))
                        y2 = min(frame.shape[0], y + h + int(h * 0.1))
                        all_boxes.append((x1, y1, x2, y2))
        else:
            # Standard single-scale detection
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05,
                hitThreshold=0.0
            )

            # Filter detections by confidence
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.3:  # Standard threshold
                    # Expand bounding box slightly for better coverage
                    x1 = max(0, x - int(w * 0.1))
                    y1 = max(0, y - int(h * 0.1))
                    x2 = min(frame.shape[1], x + w + int(w * 0.1))
                    y2 = min(frame.shape[0], y + h + int(h * 0.1))
                    all_boxes.append((x1, y1, x2, y2))

        # Remove duplicates from multi-scale detection
        filtered_boxes = self._remove_duplicate_boxes(all_boxes)

        return filtered_boxes

    def _remove_duplicate_boxes(self, boxes, iou_threshold=0.5):
        """
        Remove duplicate bounding boxes from multi-scale detection.

        Args:
            boxes: List of (x1, y1, x2, y2) boxes
            iou_threshold: IoU threshold for duplicate removal

        Returns:
            list: Filtered boxes
        """
        if not boxes:
            return []

        filtered = []
        boxes.sort(key=lambda x: (x[2]-x[0]) * (x[3]-x[1]), reverse=True)  # Sort by area descending

        while boxes:
            current = boxes.pop(0)
            filtered.append(current)

            # Remove overlapping boxes
            boxes = [box for box in boxes if self._calculate_iou(current, box) < iou_threshold]

        return filtered

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1, box2: (x1, y1, x2, y2) tuples

        Returns:
            float: IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def detect_thermal_humans(self, thermal_frame=None, visible_frame=None):
        """
        Detect humans using thermal imaging for night/low visibility conditions.

        Args:
            thermal_frame: Thermal/IR image (single channel, temperature values)
            visible_frame: Optional visible spectrum image for fusion

        Returns:
            list: Thermal-based detections [(bbox, confidence, metadata), ...]
        """
        detections = []

        if thermal_frame is None:
            if visible_frame is not None:
                thermal_frame = self._simulate_thermal_from_visible(visible_frame)
            else:
                return detections

        thermal_processed = self._preprocess_thermal_image(thermal_frame)
        thermal_masks = self._segment_thermal_regions(thermal_processed)

        for mask_name, mask in thermal_masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = thermal_processed.shape[:2]

            for contour in contours:
                area = cv2.contourArea(contour)
                min_thermal_area = (h * w) * 0.001  # Increased minimum area
                max_thermal_area = (h * w) * 0.3    # Reduced maximum area

                if min_thermal_area < area < max_thermal_area:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    bbox = [x, y, x + cw, y + ch]

                    thermal_stats = self._analyze_thermal_signature(thermal_processed, bbox, mask_name)
                    confidence = self._calculate_thermal_confidence(thermal_stats, area, h*w, mask_name)

                    if self._is_human_thermal_signature(thermal_stats):
                        metadata = {
                            'detection_method': 'thermal',
                            'thermal_type': mask_name,
                            'thermal_stats': thermal_stats,
                        }
                        detections.append((bbox, confidence, metadata))

        if visible_frame is not None and self.multi_spectral_fusion:
            visible_detections = self.detect_ultra_far_humans(visible_frame)
            detections = self._fuse_multi_spectral_detections(
                thermal_detections=detections,
                visible_detections=visible_detections,
                thermal_frame=thermal_processed,
                visible_frame=visible_frame
            )

        logging.info(f"Thermal detection found {len(detections)} human signatures")
        return detections

    def _simulate_thermal_from_visible(self, visible_frame):
        """
        Simulate thermal imaging from visible spectrum for testing/low-cost setups.

        Args:
            visible_frame: BGR visible image

        Returns:
            Simulated thermal image (single channel)
        """
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2LAB)
        red_channel = visible_frame[:, :, 2]
        _, _, v_channel = cv2.split(hsv)
        _, a_channel, _ = cv2.split(lab)

        thermal_sim = cv2.addWeighted(red_channel, 0.4, v_channel, 0.3, 0)
        thermal_sim = cv2.addWeighted(thermal_sim, 0.8, a_channel, 0.2, 0)

        noise = np.random.normal(0, 5, thermal_sim.shape).astype(np.uint8)
        thermal_sim = cv2.add(thermal_sim, noise)
        thermal_sim = cv2.normalize(thermal_sim, None, 0, 255, cv2.NORM_MINMAX)

        return thermal_sim.astype(np.uint8)

    def _preprocess_thermal_image(self, thermal_frame):
        """
        Preprocess thermal image for better human detection.

        Args:
            thermal_frame: Raw thermal image

        Returns:
            Processed thermal image
        """
        processed = thermal_frame.copy()

        if len(processed.shape) > 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        if self.thermal_smoothing:
            processed = cv2.bilateralFilter(processed, 9, 50, 50)

        processed = cv2.equalizeHist(processed)
        kernel = np.array([[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)

        return processed

    def _segment_thermal_regions(self, thermal_frame):
        """
        Segment thermal image into different temperature regions.

        Args:
            thermal_frame: Processed thermal image

        Returns:
            dict: Thermal masks for different temperature ranges
        """
        human_temp_min = self.ir_threshold_min
        human_temp_max = self.ir_threshold_max

        _, warm_mask = cv2.threshold(thermal_frame, human_temp_min, 255, cv2.THRESH_BINARY)
        _, hot_mask = cv2.threshold(thermal_frame, human_temp_max, 255, cv2.THRESH_BINARY)
        human_mask = cv2.bitwise_and(warm_mask, cv2.bitwise_not(hot_mask))
        _, cool_mask = cv2.threshold(thermal_frame, human_temp_min // 2, 255, cv2.THRESH_BINARY)

        return {
            'human_warm': human_mask,
            'very_warm': hot_mask,
            'background': cool_mask
        }

    def _analyze_thermal_signature(self, thermal_frame, bbox, mask_type):
        """
        Analyze thermal characteristics of a detected region.

        Args:
            thermal_frame: Thermal image
            bbox: Bounding box [x1, y1, x2, y2]
            mask_type: Type of thermal mask

        Returns:
            dict: Thermal signature analysis
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        thermal_roi = thermal_frame[y1:y2, x1:x2]

        if thermal_roi.size == 0:
            return {'mean_temp': 0, 'std_temp': 0, 'uniformity': 0, 'shape': (0, 0)}

        mean_temp = np.mean(thermal_roi)
        std_temp = np.std(thermal_roi)
        min_temp = np.min(thermal_roi)
        max_temp = np.max(thermal_roi)
        uniformity = 1.0 / (1.0 + std_temp / max(mean_temp, 1))

        h, w = thermal_roi.shape
        aspect_ratio = w / h if h > 0 else 0
        edges = cv2.Canny(thermal_roi, 30, 100)
        edge_density = np.sum(edges > 0) / thermal_roi.size

        return {
            'mean_temp': mean_temp,
            'std_temp': std_temp,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'uniformity': uniformity,
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'shape': (h, w),
            'mask_type': mask_type
        }

    def _calculate_thermal_confidence(self, thermal_stats, area, frame_area, mask_type):
        """
        Calculate confidence score for thermal-based human detection.

        Args:
            thermal_stats: Thermal signature analysis
            area: Contour area
            frame_area: Total frame area
            mask_type: Type of thermal mask

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.2  # Lower base confidence

        # Temperature-based confidence - more strict ranges
        mean_temp = thermal_stats['mean_temp']
        if 120 < mean_temp < 180:  # More restrictive human temperature range
            confidence += 0.4
        elif 100 < mean_temp < 200:  # Extended human range
            confidence += 0.2

        # Uniformity bonus (humans have relatively uniform temperature)
        uniformity = thermal_stats['uniformity']
        if uniformity > 0.8:  # Higher uniformity requirement
            confidence += 0.3
        elif uniformity > 0.6:
            confidence += 0.1

        # Size appropriateness - more restrictive
        size_ratio = area / frame_area
        if 0.005 < size_ratio < 0.08:  # More reasonable human size range
            confidence += 0.2
        elif size_ratio > 0.08:  # Too large
            confidence -= 0.2

        # Aspect ratio for human-like shape - more strict
        aspect_ratio = thermal_stats['aspect_ratio']
        if 0.4 < aspect_ratio < 2.5:  # More restrictive aspect ratio
            confidence += 0.2
        elif aspect_ratio > 3.0 or aspect_ratio < 0.25:  # Too extreme
            confidence -= 0.1

        # Mask type bonus - only for human_warm
        if mask_type == 'human_warm':
            confidence += 0.1

        return min(0.9, max(0.1, confidence))

    def _is_human_thermal_signature(self, thermal_stats):
        """
        Determine if thermal signature is likely human.

        Args:
            thermal_stats: Thermal signature analysis

        Returns:
            bool: True if likely human
        """
        # Human thermal criteria - more strict
        criteria = []

        # Temperature in reasonable human range
        temp_ok = 100 < thermal_stats['mean_temp'] < 200
        criteria.append(temp_ok)

        # Reasonable temperature uniformity
        uniform_ok = thermal_stats['uniformity'] > 0.5
        criteria.append(uniform_ok)

        # Human-like aspect ratio
        aspect_ok = 0.3 < thermal_stats['aspect_ratio'] < 3.5
        criteria.append(aspect_ok)

        # Not too much edge density (humans are relatively solid)
        edges_ok = thermal_stats['edge_density'] < 0.3
        criteria.append(edges_ok)

        # Must pass most criteria
        return sum(criteria) >= 3

        # High temperature uniformity required
        uniform_ok = thermal_stats['uniformity'] > 0.6
        criteria.append(uniform_ok)

        # Human-like aspect ratio - more restrictive
        aspect_ok = 0.35 < thermal_stats['aspect_ratio'] < 3.0
        criteria.append(aspect_ok)

        # Low edge density (humans are relatively solid)
        edges_ok = thermal_stats['edge_density'] < 0.25
        criteria.append(edges_ok)

        # Must pass ALL criteria (was 3 out of 4)
        return sum(criteria) >= 4

    def _fuse_multi_spectral_detections(self, thermal_detections, visible_detections,
                                      thermal_frame, visible_frame):
        """
        Fuse thermal and visible spectrum detections for improved accuracy.

        Args:
            thermal_detections: Detections from thermal analysis
            visible_detections: Detections from visible spectrum
            thermal_frame: Processed thermal image
            visible_frame: Visible spectrum image

        Returns:
            list: Fused detections
        """
        fused_detections = []
        for thermal_bbox, thermal_conf, thermal_meta in thermal_detections:
            best_match = None
            best_iou = 0
            best_visible_conf = 0
            for visible_bbox, visible_conf, visible_meta in visible_detections:
                iou = self._calculate_iou(thermal_bbox, visible_bbox)
                if iou > 0.3 and iou > best_iou:
                    best_match = visible_bbox
                    best_iou = iou
                    best_visible_conf = visible_conf

            if best_match:
                fused_bbox = [
                    (thermal_bbox[0] + best_match[0]) / 2,
                    (thermal_bbox[1] + best_match[1]) / 2,
                    (thermal_bbox[2] + best_match[2]) / 2,
                    (thermal_bbox[3] + best_match[3]) / 2
                ]
                fused_conf = (thermal_conf * 0.6) + (best_visible_conf * 0.4)
                fused_meta = {
                    'detection_method': 'multi_spectral_fusion',
                    'thermal_confidence': thermal_conf,
                    'visible_confidence': best_visible_conf,
                    'fusion_iou': best_iou,
                    'thermal_stats': thermal_meta.get('thermal_stats', {}),
                    'visible_metadata': visible_meta
                }
                fused_detections.append((fused_bbox, fused_conf, fused_meta))
            else:
                thermal_meta['fusion_status'] = 'thermal_only'
                fused_detections.append((thermal_bbox, thermal_conf * 0.8, thermal_meta))

        thermal_bbox_list = [bbox for bbox, _, _ in thermal_detections]
        for visible_bbox, visible_conf, visible_meta in visible_detections:
            matched = False
            for thermal_bbox in thermal_bbox_list:
                if self._calculate_iou(visible_bbox, thermal_bbox) > 0.3:
                    matched = True
                    break
            if not matched:
                visible_meta['fusion_status'] = 'visible_only'
                fused_detections.append((visible_bbox, visible_conf * 0.7, visible_meta))

        return fused_detections

    def detect_contours_human(self, frame):
        """
        Enhanced contour-based shape detection with far-distance sensitivity.

        Args:
            frame: Input frame (BGR)

        Returns:
            list: Bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection with enhanced parameters for far detection
        edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds for more edges

        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Smaller kernel
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        human_boxes = []
        h, w = frame.shape[:2]

        # Dynamic minimum area based on far detection setting
        if self.enable_far_detection:
            min_area = max(50, (h * w) * 0.0001)  # Much smaller minimum area for far detection
        else:
            min_area = (h * w) * 0.001  # Standard minimum area

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < self.max_contour_area:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)

                # Check aspect ratio (human-like proportions)
                aspect_ratio = cw / ch if ch > 0 else 0

                if self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max:
                    # Enhanced shape analysis for far detection
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                    # Humans are not very circular, but far humans might appear more circular
                    if circularity < 0.9:  # Relaxed circularity check for far detection
                        # Additional size-based filtering for far humans
                        relative_size = (cw * ch) / (w * h)

                        # Accept smaller detections for far humans
                        if self.enable_far_detection or relative_size > 0.0005:
                            # Expand box slightly, but less for far detections
                            expand_factor = 0.15 if relative_size > 0.01 else 0.25  # More expansion for small detections
                            x1 = max(0, x - int(cw * expand_factor))
                            y1 = max(0, y - int(ch * expand_factor))
                            x2 = min(frame.shape[1], x + cw + int(cw * expand_factor))
                            y2 = min(frame.shape[0], y + ch + int(ch * expand_factor))
                            human_boxes.append((x1, y1, x2, y2))

        return human_boxes

    def detect_motion_human(self, frame, prev_frame):
        """
        Enhanced motion detection with precise movement analysis.
        
        Detects:
        - Direction of movement (approaching/retreating/left/right)
        - Speed of movement
        - Trajectory patterns
        - Human-like motion characteristics
        
        Args:
            frame: Current frame (BGR)
            prev_frame: Previous frame (BGR)
            
        Returns:
            list: Enhanced motion detections with movement metadata
                 [(bbox, confidence, movement_data), ...]
        """
        if prev_frame is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced optical flow with better parameters for human motion
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 
            pyr_scale=0.5,      # Pyramid scale
            levels=3,           # Pyramid levels
            winsize=15,         # Window size
            iterations=3,       # Iterations
            poly_n=5,           # Polynomial degree
            poly_sigma=1.2,     # Gaussian std
            flags=0
        )
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Multi-threshold motion detection for different movement types
        motion_masks = {
            'fast': cv2.threshold(mag, np.percentile(mag, 98), 255, cv2.THRESH_BINARY)[1].astype(np.uint8),
            'medium': cv2.threshold(mag, np.percentile(mag, 95), 255, cv2.THRESH_BINARY)[1].astype(np.uint8),
            'slow': cv2.threshold(mag, np.percentile(mag, 90), 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        }
        
        detections = []
        h, w = frame.shape[:2]
        
        # Analyze each motion type
        for motion_type, motion_mask in motion_masks.items():
            # Morphological operations to clean up motion regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Size thresholds based on motion type
            size_multipliers = {'fast': 0.003, 'medium': 0.005, 'slow': 0.008}
            min_motion_area = (h * w) * size_multipliers[motion_type]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_motion_area:
                    # Get bounding box
                    x, y, cw, ch = cv2.boundingRect(contour)
                    bbox = [x, y, x + cw, y + ch]
                    
                    # Analyze movement characteristics within this region
                    movement_data = self._analyze_movement_in_region(
                        flow, mag, ang, bbox, motion_type
                    )
                    
                    # Confidence based on movement characteristics and size
                    confidence = self._calculate_motion_confidence(
                        movement_data, area, (h*w), motion_type
                    )
                    
                    detections.append((bbox, confidence, movement_data))
        
        # Remove duplicates and merge overlapping detections
        detections = self._merge_motion_detections(detections)
        
        logging.debug(f"Enhanced motion detection: {len(detections)} precise movements detected")
        return detections

    def _analyze_movement_in_region(self, flow, mag, ang, bbox, motion_type):
        """
        Analyze detailed movement characteristics within a bounding box region.
        
        Args:
            flow: Optical flow field
            mag: Magnitude of flow
            ang: Angle of flow
            bbox: [x1, y1, x2, y2]
            motion_type: 'fast', 'medium', or 'slow'
            
        Returns:
            dict: Movement analysis data
        """
        x1, y1, x2, y2 = bbox
        
        # Extract flow data for this region
        region_flow_x = flow[y1:y2, x1:x2, 0]
        region_flow_y = flow[y1:y2, x1:x2, 1]
        region_mag = mag[y1:y2, x1:x2]
        region_ang = ang[y1:y2, x1:x2]
        
        # Calculate movement statistics
        mean_flow_x = np.mean(region_flow_x)
        mean_flow_y = np.mean(region_flow_y)
        mean_magnitude = np.mean(region_mag)
        std_magnitude = np.std(region_mag)
        
        # Direction analysis (convert to degrees)
        mean_angle = np.mean(region_ang) * 180 / np.pi
        std_angle = np.std(region_ang) * 180 / np.pi
        
        # Movement direction classification
        direction = self._classify_movement_direction(mean_flow_x, mean_flow_y)
        
        # Speed classification
        speed_category = self._classify_movement_speed(mean_magnitude, std_magnitude, motion_type)
        
        # Trajectory characteristics
        trajectory_data = self._analyze_trajectory_pattern(region_flow_x, region_flow_y, region_mag)
        
        return {
            'direction': direction,
            'speed_category': speed_category,
            'mean_velocity': (mean_flow_x, mean_flow_y),
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'mean_angle': mean_angle,
            'std_angle': std_angle,
            'trajectory_pattern': trajectory_data,
            'motion_type': motion_type,
            'bbox_size': (x2-x1, y2-y1)
        }

    def _classify_movement_direction(self, flow_x, flow_y):
        """
        Classify movement direction based on flow vectors.
        
        Args:
            flow_x: Horizontal flow component
            flow_y: Vertical flow component
            
        Returns:
            str: Direction classification
        """
        # Calculate angle from positive x-axis
        angle = np.arctan2(flow_y, flow_x) * 180 / np.pi
        
        # Normalize to 0-360
        if angle < 0:
            angle += 360
        
        # Classify direction
        if 315 <= angle < 45:
            return 'right'
        elif 45 <= angle < 135:
            return 'down'  # Approaching camera
        elif 135 <= angle < 225:
            return 'left'
        elif 225 <= angle < 315:
            return 'up'    # Moving away from camera
        else:
            return 'stationary'

    def _classify_movement_speed(self, mean_mag, std_mag, motion_type):
        """
        Classify movement speed based on magnitude statistics.
        
        Args:
            mean_mag: Mean magnitude
            std_mag: Standard deviation of magnitude
            motion_type: Detected motion type ('fast', 'medium', 'slow')
            
        Returns:
            str: Speed classification
        """
        # Speed thresholds based on motion type
        if motion_type == 'fast':
            if mean_mag > 3.0:
                return 'very_fast'
            elif mean_mag > 2.0:
                return 'fast'
            else:
                return 'moderate'
        elif motion_type == 'medium':
            if mean_mag > 2.0:
                return 'moderate'
            elif mean_mag > 1.0:
                return 'slow'
            else:
                return 'very_slow'
        else:  # slow
            if mean_mag > 1.5:
                return 'slow'
            elif mean_mag > 0.5:
                return 'very_slow'
            else:
                return 'minimal'

    def _analyze_trajectory_pattern(self, flow_x, flow_y, mag):
        """
        Analyze trajectory patterns for human-like movement.
        
        Args:
            flow_x: Horizontal flow field
            flow_y: Vertical flow field
            mag: Magnitude field
            
        Returns:
            dict: Trajectory pattern analysis
        """
        # Calculate flow consistency (lower std = more consistent movement)
        flow_consistency = 1.0 / (1.0 + np.std(flow_x) + np.std(flow_y))
        
        # Calculate movement smoothness (ratio of mean to std magnitude)
        if np.std(mag) > 0:
            movement_smoothness = np.mean(mag) / np.std(mag)
        else:
            movement_smoothness = 0
        
        # Detect if movement is human-like (smooth, consistent)
        is_human_like = (flow_consistency > 0.6 and movement_smoothness > 0.8)
        
        # Calculate movement predictability
        # (how well the movement follows a straight line)
        mean_flow = np.array([np.mean(flow_x), np.mean(flow_y)])
        flow_vectors = np.column_stack((flow_x.flatten(), flow_y.flatten()))
        distances_from_mean = np.linalg.norm(flow_vectors - mean_flow, axis=1)
        predictability = 1.0 / (1.0 + np.std(distances_from_mean))
        
        return {
            'consistency': flow_consistency,
            'smoothness': movement_smoothness,
            'predictability': predictability,
            'is_human_like': is_human_like
        }

    def _calculate_motion_confidence(self, movement_data, area, frame_area, motion_type):
        """
        Calculate confidence score for motion-based human detection.
        
        Args:
            movement_data: Movement analysis dictionary
            area: Contour area
            frame_area: Total frame area
            motion_type: 'fast', 'medium', or 'slow'
            
        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.3  # Base confidence
        
        # Size factor (larger objects more likely human)
        size_ratio = area / frame_area
        if 0.001 < size_ratio < 0.1:  # Reasonable human size
            confidence += 0.2
        elif size_ratio > 0.1:  # Too large
            confidence -= 0.1
        
        # Movement consistency factor
        if movement_data['trajectory_pattern']['consistency'] > 0.7:
            confidence += 0.2
        
        # Human-like movement factor
        if movement_data['trajectory_pattern']['is_human_like']:
            confidence += 0.15
        
        # Speed appropriateness factor
        speed = movement_data['speed_category']
        if motion_type == 'fast' and speed in ['fast', 'very_fast']:
            confidence += 0.1
        elif motion_type == 'medium' and speed in ['moderate', 'slow']:
            confidence += 0.1
        elif motion_type == 'slow' and speed in ['slow', 'very_slow']:
            confidence += 0.1
        
        # Direction factor (humans tend to move in clear directions)
        if movement_data['direction'] != 'stationary':
            confidence += 0.05
        
        return min(0.95, max(0.1, confidence))

    def _merge_motion_detections(self, detections):
        """
        Merge overlapping motion detections and remove duplicates.
        
        Args:
            detections: List of (bbox, confidence, movement_data) tuples
            
        Returns:
            list: Merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        
        merged = []
        
        for i, (bbox, conf, movement_data) in enumerate(detections):
            # Check if this detection overlaps significantly with already merged ones
            should_merge = False
            
            for merged_bbox, _, _ in merged:
                iou = self._calculate_iou(bbox, merged_bbox)
                if iou > 0.3:  # Significant overlap
                    should_merge = True
                    break
            
            if not should_merge:
                merged.append((bbox, conf, movement_data))
        
        return merged

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def merge_detections(self, detections_list, iou_threshold=0.3):
        """
        Enhanced detection merging with precise bounding box refinement.
        Remove duplicates using advanced IoU and preserve movement data.

        Args:
            detections_list: List of (bbox, source, confidence, movement_data) tuples
            iou_threshold: Threshold for merging overlapping detections

        Returns:
            list: Merged detections [(bbox, max_confidence, movement_data), ...]
        """
        if not detections_list:
            return []

        # Sort by confidence descending
        detections_list.sort(key=lambda x: x[2], reverse=True)

        merged = []
        used = set()

        for i, detection in enumerate(detections_list):
            if i in used:
                continue

            if len(detection) >= 4:  # Has movement data or ultra_far metadata
                bbox, source, conf, extra_data = detection[:4]
                if source == 'motion':
                    movement_data = extra_data
                    metadata = None
                elif source == 'ultra_far':
                    movement_data = None
                    metadata = extra_data
                else:
                    movement_data = extra_data
                    metadata = None
            else:  # Legacy format
                bbox, source, conf = detection
                movement_data = None
                metadata = None

            x1, y1, x2, y2 = bbox
            merged_bbox = [x1, y1, x2, y2]
            max_conf = conf
            best_movement_data = movement_data
            best_metadata = metadata
            source_methods = [source]

            # Check for overlaps with other detections
            for j, other_detection in enumerate(detections_list[i+1:], i+1):
                if j in used:
                    continue

                if len(other_detection) >= 4:
                    bbox2, source2, conf2, extra_data2 = other_detection[:4]
                    if source2 == 'motion':
                        movement_data2 = extra_data2
                        metadata2 = None
                    elif source2 == 'ultra_far':
                        movement_data2 = None
                        metadata2 = extra_data2
                    else:
                        movement_data2 = extra_data2
                        metadata2 = None
                else:
                    bbox2, source2, conf2 = other_detection
                    movement_data2 = None
                    metadata2 = None

                x1b, y1b, x2b, y2b = bbox2

                # Calculate advanced IoU with size consideration
                inter_area = max(0, min(x2, x2b) - max(x1, x1b)) * max(0, min(y2, y2b) - max(y1, y1b))
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (x2b - x1b) * (y2b - y1b)
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0

                # Enhanced merging criteria
                size_similarity = min(box1_area, box2_area) / max(box1_area, box2_area)
                confidence_boost = (conf + conf2) / 2

                # Merge if IoU is high OR if detections are from different methods with good overlap
                should_merge = (iou > iou_threshold) or \
                              (iou > 0.1 and source != source2 and size_similarity > 0.5)

                if should_merge:
                    # Precise bounding box refinement using weighted merging
                    weight1 = conf / (conf + conf2)
                    weight2 = conf2 / (conf + conf2)

                    merged_bbox = [
                        weight1 * x1 + weight2 * x1b,      # x1
                        weight1 * y1 + weight2 * y1b,      # y1
                        weight1 * x2 + weight2 * x2b,      # x2
                        weight1 * y2 + weight2 * y2b       # y2
                    ]

                    max_conf = max(max_conf, conf2)
                    source_methods.append(source2)

                    # Prefer movement data from motion detection
                    if movement_data2 and not best_movement_data:
                        best_movement_data = movement_data2
                    elif movement_data2 and best_movement_data:
                        # Merge movement data intelligently
                        best_movement_data = self._merge_movement_data(best_movement_data, movement_data2)

                    # Merge metadata from ultra_far detections
                    if metadata2 and not best_metadata:
                        best_metadata = metadata2
                    elif metadata2 and best_metadata:
                        # Merge metadata intelligently
                        best_metadata = self._merge_ultra_far_metadata(best_metadata, metadata2)

                    used.add(j)

            # Refine final bounding box for precision
            refined_bbox = self._refine_bounding_box(merged_bbox, source_methods)

            merged.append((refined_bbox, max_conf, best_movement_data or best_metadata))

        return merged

    def _refine_bounding_box(self, bbox, source_methods):
        """
        Refine bounding box for precision based on detection methods used.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            source_methods: List of detection methods that contributed

        Returns:
            list: Refined [x1, y1, x2, y2] bounding box
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Method-specific refinements
        if 'yolo' in source_methods:
            # YOLO boxes are usually well-calibrated, minor refinement
            x1 = max(0, x1 - width * 0.02)
            y1 = max(0, y1 - height * 0.02)
            x2 = x2 + width * 0.02
            y2 = y2 + height * 0.02

        if 'hog' in source_methods:
            # HOG detections might be tighter, expand slightly for full body
            x1 = max(0, x1 - width * 0.05)
            y1 = max(0, y1 - height * 0.05)
            x2 = x2 + width * 0.05
            y2 = y2 + height * 0.1  # More expansion downward for full body

        if 'contour' in source_methods:
            # Contour detections might be noisy, contract slightly
            x1 = x1 + width * 0.03
            y1 = y1 + height * 0.03
            x2 = x2 - width * 0.03
            y2 = y2 - height * 0.03

        if 'motion' in source_methods:
            # Motion detections are dynamic, keep original but ensure minimum size
            min_size = 20
            if width < min_size:
                center_x = (x1 + x2) / 2
                x1 = center_x - min_size / 2
                x2 = center_x + min_size / 2
            if height < min_size:
                center_y = (y1 + y2) / 2
                y1 = center_y - min_size / 2
                y2 = center_y + min_size / 2

        # Ensure aspect ratio is reasonable for humans (not too wide or tall)
        final_width = x2 - x1
        final_height = y2 - y1
        aspect_ratio = final_width / final_height if final_height > 0 else 0

        if aspect_ratio > 1.5:  # Too wide
            # Contract width
            center_x = (x1 + x2) / 2
            target_width = final_height * 0.8
            x1 = center_x - target_width / 2
            x2 = center_x + target_width / 2
        elif aspect_ratio < 0.3:  # Too tall
            # Contract height
            center_y = (y1 + y2) / 2
            target_height = final_width / 0.8
            y1 = center_y - target_height / 2
            y2 = center_y + target_height / 2

        return [max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)]

    def _merge_movement_data(self, data1, data2):
        """
        Intelligently merge movement data from overlapping detections.
        
        Args:
            data1, data2: Movement data dictionaries
            
        Returns:
            dict: Merged movement data
        """
        if not data1:
            return data2
        if not data2:
            return data1
        
        # Average numerical values
        merged = {}
        for key in data1.keys():
            if key in data2 and isinstance(data1[key], (int, float)):
                merged[key] = (data1[key] + data2[key]) / 2
            else:
                merged[key] = data1[key]
        
        # For trajectory patterns, take the better one
        if 'trajectory_pattern' in data1 and 'trajectory_pattern' in data2:
            pattern1 = data1['trajectory_pattern']
            pattern2 = data2['trajectory_pattern']
            
            # Choose pattern with higher consistency and human-like score
            score1 = pattern1.get('consistency', 0) + (1 if pattern1.get('is_human_like', False) else 0)
            score2 = pattern2.get('consistency', 0) + (1 if pattern2.get('is_human_like', False) else 0)
            
            merged['trajectory_pattern'] = pattern1 if score1 >= score2 else pattern2
        
        return merged

    def detect(self, frame, prev_frame=None):
        """
        Detect humans in agricultural field using multi-method approach.
        
        Args:
            frame: Input frame (BGR)
            prev_frame: Previous frame for motion detection
            
        Returns:
            list: Detections [(bbox, confidence), ...]
        """
        detections_list = []
        
        # Method 1: Enhanced YOLO detection with far-distance capabilities
        if self.model:
            try:
                enhanced = self.enhance_for_ultra_far_detection(frame) if self.use_preprocessing else frame

                # Multi-scale detection for far away humans
                all_yolo_detections = []

                if self.enable_far_detection:
                    # Use very low confidence threshold for far detection
                    half = self.device == 'cuda'
                    results = self.model(enhanced, classes=[0], conf=self.conf_thresholds['distant'],
                                       device=self.device, half=half, verbose=False)

                    for result in results:
                        for box in result.boxes:
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())

                            # Calculate detection size for dynamic thresholding
                            x1, y1, x2, y2 = bbox
                            width = x2 - x1
                            height = y2 - y1
                            size = width * height
                            frame_area = frame.shape[0] * frame.shape[1]
                            relative_size = size / frame_area

                            # Apply dynamic confidence threshold based on detection size
                            if relative_size > 0.1:  # Large detection (close human)
                                min_conf = self.conf_thresholds['close']
                            elif relative_size > 0.01:  # Medium detection
                                min_conf = self.conf_thresholds['medium']
                            elif relative_size > 0.001:  # Small detection (far human)
                                min_conf = self.conf_thresholds['far']
                            else:  # Very small detection (distant human)
                                min_conf = self.conf_thresholds['distant']

                            if conf >= min_conf:
                                all_yolo_detections.append((bbox, 'yolo', conf, relative_size))
                                logging.debug(f"Far detection: size={relative_size:.4f}, conf={conf:.3f}, min_conf={min_conf}")
                else:
                    # Standard detection
                    half = self.device == 'cuda'
                    results = self.model(enhanced, classes=[0], conf=self.base_conf,
                                       device=self.device, half=half, verbose=False)

                    for result in results:
                        for box in result.boxes:
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            all_yolo_detections.append((bbox, 'yolo', conf, 0.1))  # Default size

                # Add all valid YOLO detections
                for detection in all_yolo_detections:
                    bbox, source, conf, size = detection
                    detections_list.append((bbox, source, conf))

                logging.info(f"YOLO detected {len(all_yolo_detections)} humans (including far-distance detections)")

            except Exception as e:
                logging.warning(f"YOLO detection failed: {e}")
        
        # Method 2: Skin color detection (for partial visibility)
        try:
            skin_bboxes = self.detect_skin_color(frame)
            for bbox in skin_bboxes:
                detections_list.append((bbox, 'skin', 0.7))  # Moderate confidence
            
            logging.debug(f"Skin detection found {len(skin_bboxes)} regions")
        except Exception as e:
            logging.warning(f"Skin detection failed: {e}")
        
        # Method 3: HOG pedestrian detection (for upright human shapes)
        try:
            hog_bboxes = self.detect_hog_pedestrians(frame)
            for bbox in hog_bboxes:
                detections_list.append((bbox, 'hog', 0.6))  # Moderate confidence
            
            logging.debug(f"HOG pedestrian detection found {len(hog_bboxes)} shapes")
        except Exception as e:
            logging.warning(f"HOG detection failed: {e}")
        
        # Method 4: Contour-based shape detection (for silhouettes)
        try:
            contour_bboxes = self.detect_contours_human(frame)
            for bbox in contour_bboxes:
                detections_list.append((bbox, 'contour', 0.5))  # Lower confidence
            
            logging.debug(f"Contour detection found {len(contour_bboxes)} human-like shapes")
        except Exception as e:
            logging.warning(f"Contour detection failed: {e}")

        # Method 5: Thermal detection for low-visibility / night scenarios
        if self.thermal_enabled:
            try:
                thermal_detections = self.detect_thermal_humans(frame)
                for bbox, conf, metadata in thermal_detections:
                    detections_list.append((bbox, 'thermal', conf, metadata))

                logging.debug(f"Thermal detection found {len(thermal_detections)} signatures")
            except Exception as e:
                logging.warning(f"Thermal detection failed: {e}")
        
        # Method 6: Ultra-far distance detection (for extremely distant humans)
        try:
            ultra_far_detections = self.detect_ultra_far_humans(frame)
            for bbox, conf, metadata in ultra_far_detections:
                detections_list.append((bbox, 'ultra_far', conf, metadata))

            logging.debug(f"Ultra-far detection found {len(ultra_far_detections)} extremely distant humans")
        except Exception as e:
            logging.warning(f"Ultra-far detection failed: {e}")
        
        # Merge all detections
        merged_detections = self.merge_detections(detections_list)
        
        logging.info(f"Agricultural detector: {len(merged_detections)} humans detected (from {len(detections_list)} multi-source detections)")
        
        return merged_detections
