import numpy as np
import cv2
import random

cap = cv2.VideoCapture(0)
img = cv2.imread('E:/m1/ti/TD7/images/flocon.png', cv2.IMREAD_UNCHANGED)  # Charger l'image du flocon
face_cascade = cv2.CascadeClassifier('E:/m1/ti/TD7/haarcascade_frontalface_alt.xml')

#Initialisation des flocons de neige
nb_flocon = 50
flocons = [[random.randint(0, 640), random.randint(0, 480)] for _ in range(nb_flocon)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Création d'un masque pour exclure les visages
    masque_visage = np.zeros_like(frame[:, :, 0])
    for (x, y, w, h) in faces:
        cv2.rectangle(masque_visage, (x, y), (x + w, y + h), 255, -1)

    for i in range(len(flocons)):
        flocons[i][1] += random.randint(2, 5)  #Descente aléatoire des flocons
        if flocons[i][1] > frame.shape[0]:
            flocons[i] = [random.randint(0, frame.shape[1]), 0]

        #Redimension du flocon
        flocon_resized = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        flocon_alpha = flocon_resized[:, :, 3] / 255.0  #Alpha du flocon
        inv_alpha = 1.0 - flocon_alpha

        x, y = flocons[i]
        h, w = flocon_resized.shape[:2]

        #Vérification si le flocon n'est pas sur un visage
        if y + h < frame.shape[0] and x + w < frame.shape[1] and masque_visage[y:y+h, x:x+w].sum() == 0:
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    flocon_alpha * flocon_resized[:, :, c] +
                    inv_alpha * frame[y:y+h, x:x+w, c]
                )

    cv2.imshow('Flocon de neige tombant derrière le visage', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
