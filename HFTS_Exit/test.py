import cv2


cap = cv2.VideoCapture(0)
if cap.isOpened():
    print(f"Camera index is available.")