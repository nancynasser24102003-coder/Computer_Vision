import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
BackGround = None

if not cap.isOpened():
    raise RuntimeError("The Camera Can not Open")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_mirror = cv2.flip(frame, 1)

    H, W = frame_mirror.shape[:2]
    Cx, Cy = W // 2, H // 2

    CurrentFrame = frame_mirror.copy()

    CurrentFrame_LAB = cv2.cvtColor(CurrentFrame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(CurrentFrame_LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    L_enh = clahe.apply(L)

    CurrentFrame_clahe = cv2.merge([L_enh, A, B])
    CurrentFrame_clahe = cv2.cvtColor(CurrentFrame_clahe, cv2.COLOR_LAB2BGR)

    Target_Mask = np.zeros((H, W), dtype=np.uint8)

    Head_Radius = 80
    Head_Center = (Cx, 100)

    cv2.circle(Target_Mask, Head_Center, Head_Radius, 255, -1)

    cv2.line(Target_Mask, (Cx, 180), (Cx, 350), 255, 100)

    cv2.line(Target_Mask, (Cx, 220), (100, 270), 255, 100)
    cv2.line(Target_Mask, (Cx, 220), (W - 100, 270), 255, 100)

    cv2.line(Target_Mask, (Cx, 350), (150, H - 10), 255, 100)
    cv2.line(Target_Mask, (Cx, 350), (W - 150, H - 10), 255, 100)

    Target_Overlay = frame_mirror.copy()
    Target_Overlay[Target_Mask > 0] = (0, 255, 0)

    frame_mirror = cv2.addWeighted(frame_mirror, 0.8, Target_Overlay, 0.2, 0)

    if BackGround is not None:

        Difference = cv2.absdiff(CurrentFrame_clahe, BackGround)
        Difference_Gray = cv2.cvtColor(Difference, cv2.COLOR_BGR2GRAY)

        T = 30
        T, Difference_Mask = cv2.threshold(Difference_Gray, T, 255, cv2.THRESH_BINARY)

        Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        Opened = cv2.morphologyEx(Difference_Mask, cv2.MORPH_OPEN, Kernel)
        Clean_Mask = cv2.morphologyEx(Opened, cv2.MORPH_CLOSE, Kernel)

        Overlap = cv2.bitwise_and(Clean_Mask, Target_Mask)

        Score = (cv2.countNonZero(Overlap) / cv2.countNonZero(Target_Mask)) * 100

        cv2.putText(frame_mirror, f"Score: {Score:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


   
    cv2.imshow("Camera", frame_mirror)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        break

    if key == ord('b'):
        BackGround = CurrentFrame_clahe.copy()

cap.release()
cv2.destroyAllWindows()