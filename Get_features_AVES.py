import os
from utils import *

max_length=3
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize an AVES classifier with 10 target classes
model = AvesClassifier( model_path='./aves-base-bio.pt').to(device)

model.eval()

data_path='/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'

audio_embeddings={'train':{},'val':{},'test':{}}
for folder_type in os.listdir(data_path):
    for category in os.listdir(os.path.join(data_path,folder_type)):
        for name_audio in os.listdir(os.path.join(data_path,folder_type,category)):
            audio_path=os.path.join(data_path,folder_type,category,name_audio)
            audio_embeddings[folder_type][name_audio]=(category,AVES_features_audio(audio_path,model,max_length,device))

# Save dictionaries to a .pt file
save_path = 'AVES_audio_embeddings.pt'
torch.save({'audio_embeddings': audio_embeddings}, save_path)
print(f"Embeddings saved to {save_path}")