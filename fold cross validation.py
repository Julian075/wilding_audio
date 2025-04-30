# main.py
from transformers import ClapModel, ClapProcessor
from model.utils import train_audio_model
from torch import load

# Parámetros de configuración
data_path = '/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'
file_path = 'Biolingual_audio_text_embeddings_4folds.pt'
num_classes = 4
k_shots = 1
batch_size = 5
num_epochs = 40
lr = 0.06174953111788025
momentum = 0.8313036304722377
weight_decay = 0.0019707582560903837
temperature = 3.69278949856813
alpha = 0.4909150213230252
temperature_closs = 0.2220842437922527
exp_name = 'Best/Final_model'

# Inicializar y entrenar con validación cruzada
model_train = train_audio_model(data_path, file_path, num_classes, k_shots, batch_size, num_epochs, lr, momentum,
                                weight_decay, temperature, temperature_closs, alpha)
cv_metrics = model_train.train_cross_validation(exp_name)

print("Cross-validation metrics:")
for fold_idx, metrics in enumerate(cv_metrics):
    print(f"Fold {fold_idx + 1}: {metrics}")
