import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread('E:/m1/ti/TD7/images/singe.png', cv2.IMREAD_UNCHANGED)
face_cascade = cv2.CascadeClassifier('E:/m1/ti/TD7/haarcascade_frontalface_alt.xml')

scale_factor = 1.5  #Ajustez cette valeur pour agrandir/réduire l'image du filtre

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        #Calcul des nouvelles dimensions du filtre
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        x_offset, y_offset = x - (new_w - w) // 2, y - (new_h - h) // 2

        #Redimension du filtre
        filtre_resize = cv2.resize(img, (new_w, new_h))

        #Gérer les bords pour éviter des dépassements
        y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
        x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)

        filtre_y1, filtre_y2 = 0, y2 - y1
        filtre_x1, filtre_x2 = 0, x2 - x1

        #Superposage du filtre avec transparence
        if filtre_resize.shape[2] == 4:
            alpha = filtre_resize[filtre_y1:filtre_y2, filtre_x1:filtre_x2, 3] / 255.0  # Couche alpha normalisée
            for c in range(3):  #Boucle sur les canaux BGR
                frame[y1:y2, x1:x2, c] = (
                    alpha * filtre_resize[filtre_y1:filtre_y2, filtre_x1:filtre_x2, c] +
                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                )

    cv2.imshow('Filtre tête de singe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
