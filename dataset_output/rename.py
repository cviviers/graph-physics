import os
import pandas as pd

# Chemin vers le fichier CSV et le dossier contenant les fichiers .npz
csv_path = "metadata.csv"
npz_folder = "latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"

# Lire le fichier CSV
metadata = pd.read_csv(csv_path)

# Parcourir chaque ligne du CSV
for file in os.listdir(npz_folder):
    if file.endswith(".npz"):
        # Obtenir le nom du fichier sans l'extension .npz
        new_name = file.replace(".obj", "")

        # Construire le chemin complet vers le fichier .npz
        old_file_path = os.path.join(npz_folder, file)
        new_file_path = os.path.join(npz_folder, new_name)

        # Renommer le fichier
        os.rename(old_file_path, new_file_path)
