
from msclap import CLAP
from utils import *

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader


root_dir = "/home/julian/PycharmProjects/pythonProject/datos/Kale/data/train"
path_info = "/home/julian/PycharmProjects/pythonProject/datos/Kale/species_frequency_summary.cvs"

# Generate captions for each species
# templ,index = templates(pd.read_csv(csv_path),species)

path='/home/julian/PycharmProjects/pythonProject/datos/weights_clap/CLAP_weights_2023.pth'
species=os.listdir(root_dir)
df_fre_info=pd.read_csv(path_info)
class_labels, index = templates(df_fre_info,species)
#class_labels, index = templates2(species)
# Create the dataset
dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index, max_length=3,model='clap')

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load and initialize CLAP
clap_model = CLAP(path,version = '2023', use_cuda=True)

#optimizer = torch.optim.Adam(clap_model.parameters(), lr=0.001)
CLAP_predict(dataloader,clap_model,class_labels)
#train_CLAP(dataloader, clap_model, class_labels, optimizer, num_epochs=10)

