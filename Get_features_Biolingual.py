from utils import *
from transformers import ClapModel, ClapProcessor
import os
import torch
import random
import numpy as np

# Configuración del modelo CLAP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")

# Parámetros de configuración
max_length = 3
data_path = '/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'
max_samples_per_category = 10  # Cambia este valor según lo necesites
num_classes = 4

# Inicialización de embeddings
audio_embeddings = {'train': {}, 'val': {}, 'test': {}}
text_embeddings = {}
id = 0


# Configuración de la semilla para replicabilidad
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Procesamiento de datos
for folder_type in os.listdir(data_path):
    for category in os.listdir(os.path.join(data_path, folder_type)):
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
        audio_files = os.listdir(os.path.join(data_path, folder_type, category))

        # Equilibrar categorías en el conjunto `train`
        if folder_type == 'train' and max_samples_per_category is not None:
            random.shuffle(audio_files)
            audio_files = audio_files[:max_samples_per_category]

        for name_audio in audio_files:
            audio_path = os.path.join(data_path, folder_type, category, name_audio)

            # Generar embedding de audio
            audio_embeddings[folder_type][name_audio] = (
                one_hot_target,
                biolingual_features_audio(audio_path, model, processor, max_length, device)
            )

# Guardar los embeddings en un archivo .pt
save_path = 'Biolingual_audio_text_embeddings2.pt'
torch.save({'audio_embeddings': audio_embeddings, 'text_embeddings': text_embeddings}, save_path)
print(f"Embeddings saved to {save_path}")


