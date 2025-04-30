import os
import random
import torch
import numpy as np
import shutil
from utils import biolingual_features_text, biolingual_features_audio
from transformers import ClapModel, ClapProcessor

# Configuración del modelo CLAP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")

# Configuración de la semilla para replicabilidad
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Parámetros de configuración
max_length = 3
data_path = '/home/julian/PycharmProjects/pythonProject/datos/Kale/data_fold/'
output_path = '/home/julian/PycharmProjects/pythonProject/datos/Kale/output_folds/'
max_samples_per_category = 10  # Cambia este valor según lo necesites
num_classes = 4

# Crear directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Inicialización de embeddings
total_folds = os.listdir(data_path)
text_embeddings = {}
id = 0

# Iterar sobre los folds para generar archivos por cada uno
for test_fold in total_folds:
    audio_embeddings = {'train': {}, 'val': {}, 'test': {}}
    test_output_path = os.path.join(output_path, test_fold)
    os.makedirs(test_output_path, exist_ok=True)

    for fold in total_folds:
        fold_path = os.path.join(data_path, fold)
        if os.path.isdir(fold_path):
            for category in os.listdir(fold_path):
                category_path = os.path.join(fold_path, category)

                # Crear directorios de categoría para train, val y test
                os.makedirs(os.path.join(test_output_path, 'train', category), exist_ok=True)
                os.makedirs(os.path.join(test_output_path, 'val', category), exist_ok=True)
                os.makedirs(os.path.join(test_output_path, 'test', category), exist_ok=True)

                # Generar embedding de texto si no existe
                if category not in text_embeddings:
                    text_embeddings[category] = {
                        'embeddings': biolingual_features_text(f'A sound of {category}', model, processor, device),
                        'id': id
                    }
                    id += 1

                # Obtener el ID de la categoría
                target = text_embeddings[category]['id']
                one_hot_target = np.zeros(num_classes)
                one_hot_target[target] = 1

                # Procesar los archivos de audio
                audio_files = os.listdir(category_path)

                # Mezclar archivos de audio
                random.shuffle(audio_files)

                for name_audio in audio_files:
                    audio_path = os.path.join(category_path, name_audio)

                    if fold == test_fold:
                        # Datos del fold actual para test
                        folder_type = 'test'
                    else:
                        # Dividir en train y val según probabilidad
                        folder_type = 'train' if random.random() < 0.8 else 'val'

                    # Equilibrar categorías en train
                    if folder_type == 'train' and max_samples_per_category is not None:
                        if len([key for key in audio_embeddings['train'] if key.startswith(category)]) >= max_samples_per_category:
                            continue

                    # Generar embedding de audio
                    audio_embeddings[folder_type][name_audio] = (
                        one_hot_target,
                        biolingual_features_audio(audio_path, model, processor, max_length, device)
                    )

                    # Copiar el archivo a la carpeta correspondiente
                    output_audio_path = os.path.join(test_output_path, folder_type, category, name_audio)
                    shutil.copy(audio_path, output_audio_path)

    # Guardar los embeddings en un archivo .pt por cada fold
    save_path = os.path.join(test_output_path, f'Biolingual_audio_text_embeddings_{test_fold}.pt')
    torch.save({'audio_embeddings': audio_embeddings, 'text_embeddings': text_embeddings}, save_path)
    print(f"Embeddings and audio files for fold {test_fold} saved to {test_output_path}")

