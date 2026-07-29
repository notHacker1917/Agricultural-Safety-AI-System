import cv2
from detection import ObjectDetector

def test_preprocessing():
    """
    Test detection with and without preprocessing.
    """
    # Load a test image (assume webcam capture or dummy)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image")
        return
    
    # Test without preprocessing
    detector_no_prep = ObjectDetector(use_preprocessing=False)
    detections_no_prep = detector_no_prep.detect(frame)
    
    # Test with preprocessing
    detector_with_prep = ObjectDetector(use_preprocessing=True)
    detections_with_prep = detector_with_prep.detect(frame)
    
    print(f"Without preprocessing: {len(detections_no_prep)} detections")
    print(f"With preprocessing: {len(detections_with_prep)} detections")
    
    # Show preprocessed image
    preprocessed = detector_with_prep.preprocessor.preprocess(frame)
    cv2.imshow('Original', frame)
    cv2.imshow('Preprocessed', preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_preprocessing()