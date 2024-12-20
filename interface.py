import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import random

root_path = os.getcwd()

sepia_set = False
def sepia():
    global sepia_set
    if not sepia_set:
        sepia_set = True
    else:
        sepia_set = False

sup1_set = False
def sup1():
    global sup1_set
    if not sup1_set:
        sup1_set = True
    else:
        sup1_set = False

sup2_set = False
def sup2():
    global sup2_set
    if not sup2_set:
        slider_visgae.grid(row=5, column=1, sticky="w")
        sup2_set = True
    else:
        sup2_set = False
        slider_visgae.grid_forget()

bg_anim_set = False
def bg_anim():
    global bg_anim_set
    if not bg_anim_set:
        bg_anim_set = True
        slider_flocon.grid(row=6, column=1, sticky="w")
    else:
        bg_anim_set = False
        slider_flocon.grid_forget()

bg_custom_set = False
def bg_custom():
    global bg_custom_set
    if not bg_custom_set:
        bg_custom_set = True
        slider_bg.grid(row=7, column=1, sticky="w")
    else:
        bg_custom_set = False
        slider_bg.grid_forget()

img_eyes = cv2.imread(f"{root_path}/images/lunettes.png", cv2.IMREAD_UNCHANGED)
face_cascade = cv2.CascadeClassifier(f"{root_path}/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(f"{root_path}/haarcascade_eye_tree_eyeglasses.xml")

img_face = cv2.imread(f"{root_path}/images/singe.png", cv2.IMREAD_UNCHANGED)
def set_img_face():
    global img_face
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=root_path)
    if file_path:
        img_face = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)


img_bg = cv2.imread(f"{root_path}/images/ville.jpg")
def set_bg():
    global img_bg
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=root_path)
    if file_path:
        img_bg = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

img_icons = cv2.imread(f"{root_path}/images/flocon.png", cv2.IMREAD_UNCHANGED)  # Charger l'image du flocon
def set_icons():
    global img_icons
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=root_path)
    if file_path:
        img_icons = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

scale_factor = 0.1
def set_scale(valeur):
    global slider_visgae
    global scale_factor
    scale_factor = float(valeur)

nb_flocon = 50
def set_flocon(valeur):
    global slider_flocon
    global nb_flocon
    nb_flocon = int(valeur)

seuil_blanc = 50  #Seuil pour détecter le fond blanc
def set_bg_value(valeur):
    global slider_bg
    global seuil_blanc
    seuil_blanc = int(valeur)

# Fonction pour mettre à jour le flux vidéo
def update_video():
    global seuil_blanc
    global scale_factor
    global img_bg
    global img_icons
    global img_eyes
    global img_face
    global face_cascade
    global eye_cascade
    global nb_flocon

    # Lire une image depuis la webcam
    ret, frame = cap.read()
    if ret:
        
        if sepia_set:
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(frame, sepia_kernel)
            frame = np.clip(sepia_image, 0, 255).astype(np.uint8)

        if sup1_set:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                eyes = eye_cascade.detectMultiScale(gray)

                #Placer les lunettes si les yeux sont détectés
                if len(eyes) >= 2:
                    #Redimension des lunettes
                    lunettes = cv2.resize(img_eyes, (w , h // 4)) 
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

        if sup2_set: 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                #Calcul des nouvelles dimensions du filtre
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                x_offset, y_offset = x - (new_w - w) // 2, y - (new_h - h) // 2

                #Redimension du filtre
                filtre_resize = cv2.resize(img_face, (new_w, new_h))

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

        if bg_anim_set:
                flocons = [[random.randint(0, 640), random.randint(0, 480)] for _ in range(nb_flocon)]

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
                    flocon_resized = cv2.resize(img_icons, (20, 20), interpolation=cv2.INTER_AREA)
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

        if bg_custom_set:
                #Redimension de l'arrière-plan à la taille de la vidéo
                bg_resize = cv2.resize(img_bg, (frame.shape[1], frame.shape[0]))

                #Convertir en HSV pour faciliter la détection du blanc
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                masque_blanc = cv2.inRange(hsv_frame, (0, 0, seuil_blanc), (180, 30, 255))

                #Inversion du masque pour isoler le sujet
                masque_sujet = cv2.bitwise_not(masque_blanc)

                #Appliquer les masques
                sujet = cv2.bitwise_and(frame, frame, mask=masque_sujet)
                fond_remplace = cv2.bitwise_and(bg_resize, bg_resize, mask=masque_blanc)

                # Fusion des deux images
                frame = cv2.add(sujet, fond_remplace)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(img)

        # Mettre à jour l'image dans le label
        label.img_tk = img_tk
        label.config(image=img_tk)

    # Appeler cette fonction après 10 ms pour continuer la mise à jour
    label.after(10, update_video)

# Interface Tkinter
root = tk.Tk()
root.title("Flux Vidéo OpenCV dans Tkinter")

# Initialiser la webcam
cap = cv2.VideoCapture(f"{root}/video.mkv")
#cap = cv2.VideoCapture(0)

# Creation de la frame
frame_options = tk.Frame(root)
frame_options.grid(row=0, column=0)

frame_video = tk.Frame(root)
frame_video.grid(row=1, column=0, columnspan=4)

# Widget Label pour afficher la vidéo
label = tk.Label(frame_video)
label.pack()

# Démarrer la mise à jour du flux vidéo
# update_video()

# Quitter proprement
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Options

sup2_btn = tk.Button(frame_options, text="Charger un visage", command=set_img_face)
sup2_btn.pack(side=tk.LEFT)

bg_btn = tk.Button(frame_options, text="Charger un fond", command=set_bg)
bg_btn.pack(side=tk.LEFT)

icons_btn = tk.Button(frame_options, text="Charger des flocons", command=set_icons)
icons_btn.pack(side=tk.LEFT)

# Checkboxes

checkbox1 = tk.Checkbutton(root, text="Activer Sépia", command=sepia)
checkbox1.grid(row=2, column=0, sticky="e")

checkbox2 = tk.Checkbutton(root, text="Image superposée 1", command=sup1)
checkbox2.grid(row=3, column=0, sticky="e")

checkbox3 = tk.Checkbutton(root, text="Image superposée 2", command=sup2)
checkbox3.grid(row=4, column=0, sticky="e")

# Barre coulissante pour varier la taille de l'image
slider_visgae = tk.Scale(root, from_=0.1, to=2,resolution=0.1, orient="horizontal", length=100, command=set_scale)
slider_visgae.set(1.5) # Valeur par défaut

slider_flocon = tk.Scale(root, from_=1, to=100, orient="horizontal", length=100, command=set_flocon)
slider_flocon.set(50) # Valeur par défaut

slider_bg = tk.Scale(root, from_=1, to=200, orient="horizontal", length=100, command=set_bg_value)
slider_bg.set(50) # Valeur par défaut

checkbox3.grid(row=5, column=0, sticky="e")


checkbox4 = tk.Checkbutton(root, text="Fond animé", command=bg_anim)
checkbox4.grid(row=6, column=0, sticky="e")

checkbox5 = tk.Checkbutton(root, text="Fond personnalisé", command=bg_custom)
checkbox5.grid(row=7, column=0, sticky="e")

update_video()

root.mainloop()
