"""
Test which camera indices work and save a test frame.
"""
import cv2
import numpy as np

print("Testing camera access...")

for idx in range(5):
    print(f"\nTesting camera index {idx}...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"  ✗ Camera {idx}: Not opened")
        continue
    
    ret, frame = cap.read()
    if not ret:
        print(f"  ✗ Camera {idx}: Cannot read frame")
        cap.release()
        continue
    
    print(f"  ✓ Camera {idx}: SUCCESS!")
    print(f"    Frame size: {frame.shape}")
    
    # Save test frame
    cv2.imwrite(f"camera_test_{idx}.jpg", frame)
    print(f"    Saved to: camera_test_{idx}.jpg")
    
    cap.release()
    break
else:
    print("\n✗ No camera found. Will use sample_video.mp4")
