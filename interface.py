import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import random

# Configuration de la fenêtre principale
root = tk.Tk()
root.title("CamEdit Studio")
root.geometry("1300x800")
root.configure(bg='#2b2b2b')

root.iconbitmap(f"{root}/ico.ico") 

# Style pour ttk
style = ttk.Style()
style.configure('Custom.TFrame', background='#2b2b2b')
style.configure('Custom.TCheckbutton', background='#2b2b2b', foreground='white')
style.configure('Custom.Horizontal.TScale', background='#2b2b2b')

# Variables
sepia_set = False
sup1_set = False
sup2_set = False
bg_anim_set = False
bg_custom_set = False
scale_factor = 1.5
nb_flocon = 50
seuil_blanc = 50

# Frame principale
main_frame = ttk.Frame(root, style='Custom.TFrame')
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Frame vidéo (à gauche)
video_frame = ttk.Frame(main_frame, style='Custom.TFrame')
video_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Label vidéo avec fond sombre
video_label = tk.Label(video_frame, bg='#1e1e1e')
video_label.pack(padx=10, pady=10, expand=True)

# Frame contrôles (à droite)
control_frame = ttk.Frame(main_frame, style='Custom.TFrame')
control_frame.pack(side="right", fill="y", padx=10, pady=10)

# Titre
title = tk.Label(control_frame,
                text="CamEdit Studio",
                font=("Helvetica", 20, "bold"),
                bg='#2b2b2b',
                fg='white')
title.pack(pady=20)

def sepia():
    global sepia_set
    if not sepia_set:
        sepia_set = True
    else:
        sepia_set = False

def sup1():
    global sup1_set
    if not sup1_set:
        sup1_set = True
    else:
        sup1_set = False

def sup2():
    global sup2_set
    if not sup2_set:
        label_visage.pack(pady=(10,0))
        slider_visage.pack(fill="x", pady=(0,10))
        sup2_set = True
    else:
        sup2_set = False
        label_visage.pack_forget()
        slider_visage.pack_forget()

def bg_anim():
    global bg_anim_set
    if not bg_anim_set:
        bg_anim_set = True
        label_flocon.pack(pady=(10,0))
        slider_flocon.pack(fill="x", pady=(0,10))
    else:
        bg_anim_set = False
        label_flocon.pack_forget()
        slider_flocon.pack_forget()


def bg_custom():
    global bg_custom_set
    if not bg_custom_set:
        bg_custom_set = True
        label_bg.pack(pady=(10,0))
        slider_bg.pack(fill="x", pady=(0,10))
    else:
        bg_custom_set = False
        label_bg.pack_forget()
        slider_bg.pack_forget()

# Fonctions de callback
def set_img_face():
    global img_face
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=f"{root_path}/images")
    if file_path:
        img_face = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

def set_bg():
    global img_bg
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=f"{root_path}/images")
    if file_path:
        img_bg = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

def set_icons():
    global img_icons
    file_path = filedialog.askopenfilename(title="Ouvrir un fichier",initialdir=f"{root_path}/images")
    if file_path:
        img_icons = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

def set_scale(valeur):
    global slider_visgae
    global scale_factor
    scale_factor = float(valeur)

def set_flocon(valeur):
    global slider_flocon
    global nb_flocon
    nb_flocon = int(float(valeur))

def set_bg_value(valeur):
    global slider_bg
    global seuil_blanc
    seuil_blanc = int(float(valeur))

# Frame pour les boutons
buttons_frame = ttk.Frame(control_frame, style='Custom.TFrame')
buttons_frame.pack(fill="x", padx=20, pady=10)

# Boutons stylisés
button_configs = [
    ("Charger un visage", '#2D5A27', set_img_face),
    ("Charger un fond", '#2D4A5A', set_bg),
    ("Charger des flocons", '#5A2D57', set_icons)
]

for text, color, command in button_configs:
    btn = tk.Button(
        buttons_frame,
        text=text,
        command=command,
        bg=color,
        fg='white',
        relief='flat',
        activebackground=color,
        activeforeground='white',
        font=("Helvetica", 10),
        pady=5
    )
    btn.pack(fill="x", pady=5)

# Séparateur
ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=20)

# Frame pour les contrôles
controls_frame = ttk.Frame(control_frame, style='Custom.TFrame')
controls_frame.pack(fill="x", padx=20, pady=10)

# Checkboxes stylisés
checkbox_configs = [
    ("Effet Sépia", sepia),
    ("Lunettes", sup1),
    ("Visage", sup2),
    ("Fond animé", bg_anim),
    ("Fond personnalisé", bg_custom)
]

for text, var in checkbox_configs:
    cb = ttk.Checkbutton(
        controls_frame,
        text=text,
        command=var,
        style='Custom.TCheckbutton'
    )
    cb.pack(fill="x", pady=5)

# Sliders avec labels
label_visage = tk.Label(controls_frame, text="Taille du visage", bg='#2b2b2b', fg='white')
slider_visage = ttk.Scale(controls_frame, from_=0.1, to=2, orient="horizontal", command=set_scale, style='Custom.Horizontal.TScale')
slider_visage.set(1.5) # Valeur par défaut

label_flocon = tk.Label(controls_frame, text="Nombre de flocons", bg='#2b2b2b', fg='white')
slider_flocon = ttk.Scale(controls_frame, from_=1, to=100, orient="horizontal", command=set_flocon, style='Custom.Horizontal.TScale')
slider_flocon.set(50) # Valeur par défaut

label_bg = tk.Label(controls_frame, text="Seuil du fond", bg='#2b2b2b', fg='white')
slider_bg = ttk.Scale(controls_frame, from_=1, to=200, orient="horizontal", command=set_bg_value, style='Custom.Horizontal.TScale')
slider_bg.set(50) # Valeur par défaut


# Initialisation des ressources
root_path = os.getcwd()
img_eyes = cv2.imread(f"{root_path}/images/lunettes.png", cv2.IMREAD_UNCHANGED)
img_face = cv2.imread(f"{root_path}/images/visages/singe.png", cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread(f"{root_path}/images/bg/ville.jpg")
img_icons = cv2.imread(f"{root_path}/images/icons/flocon.png", cv2.IMREAD_UNCHANGED)

face_cascade = cv2.CascadeClassifier(f"{root_path}/haar/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(f"{root_path}/haar/haarcascade_eye_tree_eyeglasses.xml")

# Fonction de mise à jour vidéo
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

        if bg_anim_set:
                flocons = [[random.randint(0, frame.shape[1]), random.randint(0,  frame.shape[0])] for _ in range(nb_flocon)]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # Création d'un masque pour exclure les visages
                masque_visage = np.zeros_like(frame[:, :, 0])
                for (x, y, w, h) in faces:
                    cv2.rectangle(masque_visage, (x-50, y-10), (x + w+50, y + h*3), 255, -1)

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

            if sepia_set:
                sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])
                sepia_image = cv2.transform(frame, sepia_kernel)
                frame = np.clip(sepia_image, 0, 255).astype(np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(img) 


        # Mettre à jour l'image dans le label
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

    # Appeler cette fonction après 10 ms pour continuer la mise à jour
    video_label.after(10, update_video)

# Configuration de la webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(f"{root}/videofond2.mp4")


# Fonction de fermeture
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Démarrage
update_video()
root.mainloop()