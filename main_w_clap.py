from transformers import ClapModel, ClapProcessor
from model.audio_model import Audio_classifier
from torch.utils.data import DataLoader
from utils import *
import torch
import os
from tqdm import tqdm
import numpy as np

# Hyperparameters
batch_size = 5
num_classes = 4
k_shots = 1  # Numbers of support examples per category
num_epochs = 40
lr = 0.06174953111788025
momentum = 0.8313036304722377
weight_decay = 0.0019707582560903837
temperature = 3.69278949856813
alpha = 0.4909150213230252
temperature_closs = 0.2220842437922527
embed_size = 512#768  # CLAP embedding size

# Load CLAP model and processor
model_id = "laion/clap-htsat-unfused"
processor = ClapProcessor.from_pretrained(model_id)
clap_model = ClapModel.from_pretrained(model_id)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clap_model = clap_model.to(device)
clap_model.eval()  # Set to evaluation mode since we'll only use it for features

# Data paths
data_path = '/home/ids/jpabon/projects/wilding_audio/data/Kale/data'
train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'val')
test_path = os.path.join(data_path, 'test')

# Get species and create templates
species = sorted(os.listdir(test_path))
class_labels = [f"This is a sound of {sp}" for sp in species]

# Create index mapping
index = {sp: i for i, sp in enumerate(species)}

def extract_clap_features(audio_path):
    audio, sr = load_aud(audio_path, max_length=3)
    audio_inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
    
    with torch.no_grad():
        audio_features = clap_model.get_audio_features(**audio_inputs)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
    
    return audio_features.cpu()

# Prepare support set (k-shot examples from training set)
support_set = []
for sp in species:
    sp_dir = os.path.join(train_path, sp)
    audio_files = sorted(os.listdir(sp_dir))[:k_shots]  # Get k examples
    for audio_file in audio_files:
        audio_path = os.path.join(sp_dir, audio_file)
        features = extract_clap_features(audio_path)
        one_hot = torch.zeros(num_classes)
        one_hot[index[sp]] = 1
        support_set.append((audio_path, (features, one_hot)))

# Create datasets
train_dataset = AudioCaptionDatasetFromFolder(root_dir=train_path, target_index=index, max_length=3, model='clap')
val_dataset = AudioCaptionDatasetFromFolder(root_dir=val_path, target_index=index, max_length=3, model='clap')
test_dataset = AudioCaptionDatasetFromFolder(root_dir=test_path, target_index=index, max_length=3, model='clap')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the audio classifier
model = Audio_classifier(num_classes, k_shots, embed_size, support_set).to(device)

# Process text prompts
text_inputs = processor(text=class_labels, return_tensors="pt", padding=True)
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
with torch.no_grad():
    text_features = clap_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

print(f"Training on {device}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {species}")

# Training loop
best_val_acc = 0
best_model_path = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Training
    for audio_paths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        # Extract CLAP features for the batch
        audio_features = torch.cat([extract_clap_features(path) for path in audio_paths]).to(device)
        
        # Forward pass
        logits = model(audio_features, text_features, temperature, alpha)
        
        # Calculate loss
        loss = model.Contrastive_loss(logits, labels.to(device), temperature_closs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()
    
    train_acc = 100 * correct / total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for audio_paths, labels, _ in val_loader:
            audio_features = torch.cat([extract_clap_features(path) for path in audio_paths]).to(device)
            logits = model(audio_features, text_features, temperature, alpha)
            _, predicted = torch.max(logits.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels.to(device)).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = f'best_model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'species': species,
            'class_labels': class_labels
        }, best_model_path)

# Test evaluation
print("\nEvaluating best model on test set...")
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
model.eval()

test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for audio_paths, labels, _ in tqdm(test_loader, desc="Testing"):
        audio_features = torch.cat([extract_clap_features(path) for path in audio_paths]).to(device)
        logits = model(audio_features, text_features, temperature, alpha)
        _, predicted = torch.max(logits.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels.to(device)).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = 100 * test_correct / test_total

# Print final results
print(f"\nTest Accuracy: {test_acc:.2f}%")

# Calculate and save confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(conf_matrix, index=species, columns=species)
print("\nConfusion Matrix:")
print(cm_df)

report = classification_report(all_labels, all_preds, target_names=species, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(report_df)

# Save results
results_dir = 'results/audio_classifier'
os.makedirs(results_dir, exist_ok=True)
cm_df.to_csv(f'{results_dir}/confusion_matrix.csv')
report_df.to_csv(f'{results_dir}/classification_report.csv')
