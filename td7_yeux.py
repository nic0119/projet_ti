import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread('E:/m1/ti/TD7/images/lunettes.png', cv2.IMREAD_UNCHANGED)
face_cascade = cv2.CascadeClassifier('E:/m1/ti/TD7/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('E:/m1/ti/TD7/haarcascade_eye_tree_eyeglasses.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        eyes = eye_cascade.detectMultiScale(gray)

        #Placer les lunettes si les yeux sont détectés
        if len(eyes) >= 2:
            #Redimension des lunettes
            lunettes = cv2.resize(img, (w , h // 4)) 
            lw, lh, _ = lunettes.shape

            #Gérer les bords pour éviter les dépassements
            y1, y2 = max(0, y + h // 4), min(frame.shape[0], y + h // 4 + lw)
            x1, x2 = max(0, x), min(frame.shape[1], x + lh)

            lunettes_y1, lunettes_y2 = 0, y2 - y1
            lunettes_x1, lunettes_x2 = 0, x2 - x1

            #Superposage du filtre avec transparence
            if lunettes.shape[2] == 4:
                alpha = lunettes[lunettes_y1:lunettes_y2, lunettes_x1:lunettes_x2, 3] / 255.0
                for c in range(3):  #Boucle sur les canaux BGR
                    frame[y1:y2, x1:x2, c] = (
                        alpha * lunettes[lunettes_y1:lunettes_y2, lunettes_x1:lunettes_x2, c] +
                        (1 - alpha) * frame[y1:y2, x1:x2, c]
                    )

    cv2.imshow('Filtre lunettes dollar', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
