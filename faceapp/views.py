import os
import cv2
import numpy as np
from django.shortcuts import render
from insightface.app import FaceAnalysis

# 🔹 Initialisation globale du moteur ArcFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 🔹 Dossier des visages connus
FACES_DIR = 'media/faces/'
os.makedirs(FACES_DIR, exist_ok=True)  # crée le dossier si nécessaire

# 🔹 Fonction pour comparer deux visages
def compare_faces(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return 0.0

    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if not faces1 or not faces2:
        return 0.0

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# 🔹 Fonction pour reconnaître un visage
def recognize_face(uploaded_img_path):
    best_score = 0
    best_name = "Inconnu ❌"

    for file_name in os.listdir(FACES_DIR):
        known_img_path = os.path.join(FACES_DIR, file_name)
        score = compare_faces(uploaded_img_path, known_img_path)
        if score > best_score:
            best_score = score
            best_name = os.path.splitext(file_name)[0]  # nom du fichier sans extension

    # seuil de décision
    if best_score < 0.3:
        best_name = "Inconnu ❌"
    return best_name, best_score

# 🔹 Vue principale
def index(request):
    result = None
    img_url = None

    if request.method == 'POST':
        img = request.FILES.get('img')
        if img:
            os.makedirs('media/uploads', exist_ok=True)
            img_path = f'media/uploads/{img.name}'
            with open(img_path, 'wb+') as f:
                for chunk in img.chunks():
                    f.write(chunk)

            # 🔍 Reconnaissance
            result, score = recognize_face(img_path)
            img_url = '/' + img_path

    # 🔹 Toujours retourner un HttpResponse
    return render(request, 'faceapp/index.html', {
        'result': result,
        'img_url': img_url,
    })
