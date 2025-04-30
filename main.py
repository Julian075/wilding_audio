
from transformers import ClapModel, ClapProcessor
from model.utils import train_audio_model
from torch import load



#load the encoder model
model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
prompt = "This is a sound of"
#define the number of batch_size
batch_size=5
#The path with all the data (train, test, val)
data_path='/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'

#Verify if the features are extracted yet
file_path = 'Biolingual_audio_text_embeddings_fold_1.pt' # 'Biolingual_audio_text_embeddings2.pt' #
num_classes=4
k_shots=1 # Numbers of support examples of each categorie
num_epochs=40
lr=0.06174953111788025
momentum=0.8313036304722377
weight_decay=0.0019707582560903837
temperature=  3.69278949856813
alpha = 0.4909150213230252
temperature_closs= 0.2220842437922527

model_train = train_audio_model( data_path, file_path, num_classes, k_shots, batch_size, num_epochs, lr, momentum, weight_decay, temperature,temperature_closs, alpha)
exp_name='Best/Final_model'
path_best_model, _ = model_train.train(exp_name)
loaded_model = load(path_best_model)
metrics = model_train.test(loaded_model)
#{'num_epochs': 40, 'batch_size': 5, 'lr': 0.06174953111788025, 'k_shots': 1, 'momentum': 0.8313036304722377,
# 'weight_decay': 0.0019707582560903837, 'temperature': 3.69278949856813, 'temperature_closs': 0.2220842437922527, 'alpha': 0.4909150213230252}
