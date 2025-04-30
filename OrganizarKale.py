import os
import shutil
import pandas as pd
from pydub import AudioSegment
import random

def cut_segment(path_aud,t1,t2,output_path):
    # Cargar el archivo de audio
    sound = AudioSegment.from_wav(path_aud)

    # Cortar el audio (en milisegundos)
    t1 = t1*1000
    t2 = t2*1000
    if t2-t1<1000:
        t2=t1+1000
    newAudio = sound[t1:t2]

    # Exportar el nuevo audio
    newAudio.export(output_path, format="wav")

point={'SMA03286':'KLSA01','SMA03294':'KLSA02','SMA03332':'KLSA03',
       'SMA03322':'KLSA04','SMA03210':'KLSA05','SMA03328':'KLSA06',
       'SMA03297':'KLSA07','SMA03175ICP':'KLSA08','SMA03247':'KLSA09',
       'SMA03327':'KLSA10','SMA03393':'KLSA11','SMA03406':'KLSA12',
       'SMA03320':'KLSA13','SMA03126':'KLSA14','SMA03326':'KLSA15',
       'SMA03330':'KLSA16','SMA03251':'KLSA17','SMA03411':'KLSA18'}


sites_train_val=['KLSA01','KLSA02','KLSA04','KLSA06','KLSA08','KLSA09','KLSA11','KLSA14',
                 'KLSA15','KLSA16','KLSA17','KLSA18']
sites_test=['KLSA03','KLSA05','KLSA07','KLSA10','KLSA12','KLSA13']

path='/home/julian/PycharmProjects/pythonProject/datos/Kale/All sounds'
df=pd.read_excel('/home/julian/PycharmProjects/pythonProject/datos/Kale/1_filtered_allFeaturesandLabels.xlsx')
df_seg_malos=pd.read_excel('/home/julian/PycharmProjects/pythonProject/datos/Kale/posibles_segmentos_malos.xlsx')
new_path='/home/julian/PycharmProjects/pythonProject/datos/Kale/data'
os.makedirs(new_path,exist_ok=True)

#lista de seg malos
seg_malos = df_seg_malos['Lonely Index'].tolist()



for _,row in df.iterrows():
    site = point[row['File'].split('_')[0]]
    t1= row['Start']
    t2 = row['End']
    ok=False
    if not(row['id'] in seg_malos):# mirar que este fuera de la lista de posibles fallas

        if site in sites_train_val:
            # Split into train and val based on a random 80/20 split
            if random.random() < 0.8:
                folder='train'
            else:
                folder='val'
            ok=True
        elif site in sites_test:
            folder='test'
            ok=True
        if ok:
            os.makedirs(os.path.join(new_path, folder,row['Specie ID']), exist_ok=True)
            out=os.path.join(new_path,folder,row['Specie ID'],str(row['id'])+'_'+row['File'])
            aud_name=os.path.join(path,row['File'])
            cut_segment(aud_name, t1, t2, out)




#for aud in os.listdir(path):
#    name=aud.split('_')[0]
#    os.makedirs(os.path.join(new_path,name), exist_ok=True)
#    shutil.copy(os.path.join(path,aud),os.path.join(new_path,name,aud))
