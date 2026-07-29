import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    print(f'index {i}: opened={ok}')
    if ok:
        ret, frame = cap.read()
        print(f'  read={ret}, size={None if frame is None else frame.shape}')
        cap.release()
