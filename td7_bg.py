import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img = cv2.imread('E:/m1/ti/TD7/images/ville.jpg')
seuil_blanc = 50  #Seuil pour détecter le fond blanc

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #Redimension de l'arrière-plan à la taille de la vidéo
    bg_resize = cv2.resize(img, (frame.shape[1], frame.shape[0]))

    #Convertir en HSV pour faciliter la détection du blanc
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masque_blanc = cv2.inRange(hsv_frame, (0, 0, seuil_blanc), (180, 30, 255))

    #Inversion du masque pour isoler le sujet
    masque_sujet = cv2.bitwise_not(masque_blanc)

    #Appliquer les masques
    sujet = cv2.bitwise_and(frame, frame, mask=masque_sujet)
    fond_remplace = cv2.bitwise_and(bg_resize, bg_resize, mask=masque_blanc)

    # Fusion des deux images
    res = cv2.add(sujet, fond_remplace)

    cv2.imshow("Fond blanc remplacé par une ville", res)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
