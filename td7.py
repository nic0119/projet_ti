import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread('E:/m1/ti/TD7/images/singe.png', cv2.IMREAD_UNCHANGED)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #Applicage du filtre sépia
    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(frame, sepia_kernel)
    frame = np.clip(sepia_image, 0, 255).astype(np.uint8)

    cv2.imshow('Filtre sepia', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()