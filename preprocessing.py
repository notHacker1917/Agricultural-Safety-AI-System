import cv2
import numpy as np
import logging

class ImagePreprocessor:
    """
    Preprocessing pipeline for agricultural environments.
    Improves detection under dust, shadows, low-light.
    """
    def __init__(self, enable_clahe=True, enable_blur=True, enable_brightness_norm=True):
        self.enable_clahe = enable_clahe
        self.enable_blur = enable_blur
        self.enable_brightness_norm = enable_brightness_norm
        logging.info("Image preprocessor initialized")

    def preprocess(self, image):
        """
        Apply preprocessing pipeline.

        Args:
            image (numpy array): BGR image.

        Returns:
            numpy array: Preprocessed image.
        """
        processed = image.copy()

        # Brightness normalization
        if self.enable_brightness_norm:
            processed = self._normalize_brightness(processed)

        # Contrast enhancement (CLAHE)
        if self.enable_clahe:
            processed = self._apply_clahe(processed)

        # Noise reduction (Gaussian blur)
        if self.enable_blur:
            processed = self._apply_gaussian_blur(processed)

        return processed

    def _normalize_brightness(self, image):
        """
        Normalize brightness using histogram equalization.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_clahe(self, image):
        """
        Apply CLAHE for contrast enhancement.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_gaussian_blur(self, image, ksize=(3, 3), sigma=0.5):
        """
        Apply Gaussian blur for noise reduction.
        """
        return cv2.GaussianBlur(image, ksize, sigma)