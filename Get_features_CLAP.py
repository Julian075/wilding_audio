from utils import *
from msclap import CLAP
import os

#CLAP model
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load and initialize CLAP
path='/home/julian/PycharmProjects/pythonProject/datos/weights_clap/CLAP_weights_2023.pth'
clap_model = CLAP(path,version = '2023', use_cuda=True)
max_length=3
data_path="/home/julian/PycharmProjects/pythonProject/datos/Kale/data"

audio_embeddings={'train':{},'val':{},'test':{}}
embeddings=[]



for folder_type in os.listdir(data_path):
    for category in os.listdir(os.path.join(data_path,folder_type)):
        if category not in embeddings:
            embeddings.append(f'A sound of {category}')
        for name_audio in os.listdir(os.path.join(data_path,folder_type,category)):
            audio_path=os.path.join(data_path,folder_type,category,name_audio)
            audio_path=(audio_path,)
            audio_embeddings[folder_type][name_audio]=clap_model.get_audio_embeddings(audio_path, resample=True)



text_embeddings= {
    'embeddings': clap_model.get_text_embeddings(embeddings)
}
# Save dictionaries to a .pt file
save_path = 'CLAP_audio_text_embeddings.pt'
torch.save({'audio_embeddings': audio_embeddings, 'text_embeddings': text_embeddings}, save_path)
print(f"Embeddings saved to {save_path}")


