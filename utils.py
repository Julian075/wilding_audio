
from transformers import ClapProcessor
from torch.utils.data import Dataset
import torch.nn.functional as F
import soundfile as sf
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import torch
import pandas as pd
import fairseq
import torch.nn as nn




## Create a mapping for captions based on the folder (species) name
def create_caption_from_species(species_name, mean_f_peak, mean_f_min, mean_f_max):
    return f"Species: {species_name}, Peak Frequency: {mean_f_peak} khz, Min Frequency: {mean_f_min} khz, Max Frequency: {mean_f_max} khz"

def templates(df,list_sp):
    templ = []
    target_index={}
    cont=0
    for index, row in df.iterrows():
        if row['species'] in list_sp:
            templ.append( create_caption_from_species(row['species'], row['mean_f_peak'], row['mean_f_min'], row['mean_f_max']))
            target_index[row['species']]=cont
            cont=cont+1
    return templ,target_index


def templates2(class_names):
    templ = []
    target_index={}
    cont=0
    for class_name in class_names:
        caption=f"A sound of {class_name}"
        templ.append( caption)
        target_index[class_name]=cont
        cont=cont+1
    return templ,target_index

def templates3(class_names):
    templ = []
    target_index={}
    cont=0
    for class_name in class_names:
        templ.append( f"{class_name}")
        target_index[class_name]=cont
        cont=cont+1
    return templ,target_index
def load_aud(audio_path,max_length):
    # Load the audio file
    audio_sample, sample_rate = sf.read(audio_path)
    # Convert max_length from seconds to samples
    max_length_in_samples = int(max_length * sample_rate)

    # Pad or truncate the audio to the desired max_length in samples
    if len(audio_sample) > max_length_in_samples:
        audio_sample = audio_sample[:max_length_in_samples]  # Truncate if too long
    elif len(audio_sample) < max_length_in_samples:
        # Pad if too short
        padding = np.zeros(max_length_in_samples - len(audio_sample))
        audio_sample = np.concatenate((audio_sample, padding))
    return audio_sample,sample_rate




class AudioCaptionDatasetFromFolder(Dataset):
    def __init__(self, root_dir, target_index=None, max_length=None,model=''):

        self.root_dir = root_dir
        self.audio_files = []
        self.targets = []
        self.index = target_index if target_index else {}
        self.max_length=max_length
        self.num_classes=len(target_index)
        self.model=model

        # Traverse the folder structure and collect file paths and species names
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path): #and species_folder in species:
                for file_name in os.listdir(species_path):
                    if file_name.endswith(".wav"):  # Assuming audio files are in .wav format
                        self.audio_files.append(os.path.join(species_path, file_name))
                        self.targets.append(self.index[species_folder])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        target = self.targets[idx]

        # Crear la codificación one-hot para el target
        one_hot_target = np.zeros(self.num_classes)
        one_hot_target[target] = 1
        if self.model == 'biolingual':
            # Load the audio file
            audio, sample_rate = load_aud(audio_path,self.max_length)
            return audio, target ,one_hot_target
        elif self.model=='clap':
            return audio_path, target, one_hot_target

class AvesClassifier(nn.Module):
    def __init__(self, model_path, num_classes=10, embeddings_dim=768, multi_label=False):

        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        out = self.model.extract_features(x)[0]
        out = out.mean(dim=1)             # mean pooling
        logits = self.head(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
    def encode_audio(self, x):
        out = self.model.extract_features(x)[0]
        out = out.mean(dim=1)             # mean pooling

        return out
def AVES_features_audio(audio_path,model,max_length,device):
    # Load the audio file
    audio_sample, _ = load_aud(audio_path, max_length)
    audio_sample = torch.from_numpy(audio_sample).float().unsqueeze(0).to(device)
    with torch.no_grad():
        audio_embed=model.encode_audio(audio_sample)
    return audio_embed

def biolingual_features_audio(audio_path,model,processor,max_length,device):
    # Load the audio file
    audio_sample, sample_rate = load_aud(audio_path,max_length)

    with torch.no_grad():
        inputs = processor(audios=audio_sample, return_tensors="pt").to(device)
        audio_embed = model.get_audio_features(**inputs)
    return audio_embed

def biolingual_features_text(prompt,model,processor,device):
    with torch.no_grad():
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        text_embed = model.get_text_features(**text_inputs)
    return text_embed

def biolingual_predict(dataloader,audio_classifier,templ):
    correct_predictions = 0
    total_predictions = 0

    y_true = []
    y_pred = []

    for audio, target, one_hot_target in dataloader:
        audio_array = []
        for i in range(len(audio)):
            audio_array.append(audio[i].numpy())  # Convert the audio sample to a NumPy array

        output = audio_classifier(audio_array, candidate_labels=templ)  # Zero-shot classification

        # Iterate over the batch
        for i in range(len(audio)):
            # Get the predicted label with the highest score
            predicted_label = max(output[i], key=lambda x: x['score'])['label']

            # Append to y_true and y_pred
            one_hot_pred = np.zeros(len(one_hot_target[0]))
            one_hot_pred[templ.index(predicted_label)] = 1
            y_pred.append(one_hot_pred)
            y_true.append(one_hot_target[i].detach().cpu().numpy())

    # Convert one-hot encoded `y_true` and `y_pred` to class indices
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc * 100:.2f}%')
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate F1-score and other metrics
    report = classification_report(y_true, y_pred, target_names=templ, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(conf_matrix, index=templ, columns=templ)

    # Save both confusion matrix and classification report to a CSV file
    combined_df = pd.concat([cm_df, report_df], axis=1)
    combined_df.to_csv('classification_report.csv', index=True)

def CLAP_predict(dataloader,clap_model,templ):
    # Computing text embeddings
    text_embeddings = clap_model.get_text_embeddings(templ)
    # Computing audio embeddings
    y_preds, y_labels = [], []
    for x, _, one_hot_target in dataloader:
        audio_embeddings = clap_model.get_audio_embeddings(x, resample=True)
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())

    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    # Convert one-hot encoded `y_labels` and probability-based `y_preds` to class indices
    y_true = np.argmax(y_labels, axis=1)
    y_pred_classes = np.argmax(y_preds, axis=1)
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred_classes)
    print(f'Accuracy: {acc * 100:.2f}%')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate F1-score and other metrics
    report = classification_report(y_true, y_pred_classes, target_names=templ, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(conf_matrix, index=templ, columns=templ)

    # Save both confusion matrix and classification report to a CSV file
    combined_df = pd.concat([cm_df, report_df], axis=1)
    combined_df.to_csv('classification_report.csv', index=True)


def train_CLAP(dataloader, clap_model, templ, optimizer, num_epochs=10):
    # Set model to training mode
    clap_model.train()

    # Compute text embeddings for labels once
    text_embeddings = clap_model.get_text_embeddings(templ)

    for epoch in range(num_epochs):
        running_loss = 0.0
        y_true, y_pred_classes = [], []

        # Iterate through batches in the dataloader
        for x, _, one_hot_target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()  # Reset gradients

            # Compute audio embeddings
            audio_embeddings = clap_model.get_audio_embeddings(x, resample=True)

            # Compute similarity between audio and text embeddings
            similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

            # Compute loss
            y_true_batch = torch.argmax(one_hot_target, dim=1).to(similarity.device)  # Convert one-hot to class indices
            loss = F.cross_entropy(similarity, y_true_batch)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

            # Store true and predicted labels for accuracy calculation
            y_true.extend(y_true_batch.cpu().numpy())
            y_pred_classes.extend(torch.argmax(similarity, dim=1).cpu().numpy())

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        # Calculate accuracy for the epoch
        epoch_acc = accuracy_score(y_true, y_pred_classes)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {epoch_acc * 100:.2f}%")

    print("Training Complete")
