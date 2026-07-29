"""
Generate a synthetic sample video with mock people for demo purposes.
"""
import cv2
import numpy as np

def generate_sample_video(output_path='sample_video.mp4', num_frames=30, width=640, height=480):
    """
    Generate a synthetic video with moving rectangles (mock people) - BRIGHT VERSION.
    
    Args:
        output_path (str): Path to save the video.
        num_frames (int): Number of frames to generate.
        width (int): Video width.
        height (int): Video height.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating bright synthetic video: {output_path}")
    
    for frame_idx in range(num_frames):
        # Create a frame with BRIGHT white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Very bright background
        
        # Add grid for reference
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)
        
        # Add bright title bar
        cv2.rectangle(frame, (0, 0), (width, 50), (50, 50, 200), -1)
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{num_frames}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some mock people (moving rectangles) - BRIGHT colors
        # Person 1 - moving left to right - BRIGHT cyan
        x1 = 50 + (frame_idx * 3)
        y1 = 100
        w1, h1 = 80, 150
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), -1)  # Cyan filled
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), 3)  # Black border
        cv2.putText(frame, "P1", (x1 + 20, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Person 2 - moving right to left - BRIGHT yellow
        x2 = 500 - (frame_idx * 2)
        y2 = 200
        w2, h2 = 70, 140
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), -1)  # Yellow filled
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 0), 3)  # Black border
        cv2.putText(frame, "P2", (x2 + 15, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Add bright hazard zone - RED circle
        cv2.circle(frame, (320, 380), 60, (0, 0, 255), -1)  # Red filled
        cv2.circle(frame, (320, 380), 60, (255, 255, 255), 3)  # White border
        cv2.putText(frame, "HAZARD", (270, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add coordinates display
        cv2.putText(frame, f"X: {frame_idx * 10}  Y: {frame_idx * 5}", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        out.write(frame)
        print(f"  Frame {frame_idx + 1}/{num_frames}")
    
    out.release()
    print(f"✓ BRIGHT video saved to: {output_path}")

if __name__ == '__main__':
    generate_sample_video('sample_video.mp4', num_frames=30)
