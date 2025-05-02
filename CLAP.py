from transformers import ClapModel, ClapProcessor
from utils import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import soundfile as sf

root_dir = "/home/ids/jpabon/projects/wilding_audio/data/Kale/data/test"

# Load CLAP model and processor from Hugging Face
model_id = "laion/clap-htsat-unfused"  # This is the latest CLAP model
processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get species and create simple templates
species = sorted(os.listdir(root_dir))
class_labels = [f"This is a sound of {sp}" for sp in species]

# Create index mapping
index = {sp: i for i, sp in enumerate(species)}

# Create the dataset
dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index, max_length=3, model='clap')

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def load_audio(audio_path, sampling_rate=48000):
    """Load an audio file and resample it."""
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    return audio, sr

def CLAP_predict(dataloader, model, processor, class_labels, species, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Process text inputs once
    text_inputs = processor(text=class_labels, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        # Get text embeddings
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Process audio files
        for audio_paths, labels, one_hot_target in tqdm(dataloader, desc="Evaluating"):
            # Load and process audio
            audio_path = audio_paths[0]  # Get the first path since batch_size=1
            audio, sr = load_audio(audio_path)
            
            # Process audio with the CLAP processor
            audio_inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
            
            # Get audio embeddings
            audio_features = model.get_audio_features(**audio_inputs)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = torch.matmul(audio_features, text_features.T)
            predictions = similarity.softmax(dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(one_hot_target.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    y_true = np.argmax(all_labels, axis=1)
    y_pred = np.argmax(all_preds, axis=1)
    
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc * 100:.2f}%')
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(conf_matrix, index=species, columns=species)
    print(cm_df)
    
    # Calculate classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=species, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    
    # Save results
    results_dir = 'results/clap'
    os.makedirs(results_dir, exist_ok=True)
    cm_df.to_csv(f'{results_dir}/confusion_matrix.csv')
    report_df.to_csv(f'{results_dir}/classification_report.csv')

# Run evaluation
print(f"Using device: {device}")
print(f"Found {len(dataset)} samples in {len(species)} classes")
print(f"Classes: {species}")
print("Using prompt template: 'This is a sound of [species]'")
CLAP_predict(dataloader, model, processor, class_labels, species, device)

