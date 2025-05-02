import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Load the embeddings
embeddings_path = './Pret_Weights_and_Features/AVES_audio_embeddings.pt'
data = torch.load(embeddings_path)
audio_embeddings = data['audio_embeddings']

# Get all unique categories
categories = set()
for split in ['train', 'val', 'test']:
    for audio_name, (category, _) in audio_embeddings[split].items():
        categories.add(category)
categories = sorted(list(categories))

# Create category to index mapping
cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

#Prepare data
def prepare_data(split):
    X = []
    y = []
    for audio_name, (category, embedding) in audio_embeddings[split].items():
        X.append(embedding.squeeze().cpu().numpy())  # Asegurarnos de que el embedding esté en la forma correcta
        y.append(cat_to_idx[category])
    return np.array(X), np.array(y)

# Prepare train, val and test sets
X_train, y_train = prepare_data('train')
X_val, y_val = prepare_data('val')
X_test, y_test = prepare_data('test')

# Convert to torch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Define the classifier
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Asegurarnos de que x tiene la forma correcta
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Eliminar dimensión extra si existe
        return self.classifier(x)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]  # Ahora debería ser 768
num_classes = len(categories)
model = AudioClassifier(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 32

# Training loop
best_val_acc = 0
patience = 10
patience_counter = 0

print(f"Training on {device}")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of test samples: {len(X_test)}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {categories}")
print(f"Input dimension: {input_dim}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_acc = 100 * correct / total
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_loss = criterion(val_outputs, y_val.to(device))
        _, predicted = torch.max(val_outputs.data, 1)
        val_acc = 100 * (predicted == y_val.to(device)).sum().item() / len(y_val)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {total_loss/len(X_train):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_aves_classifier.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load best model for testing
model.load_state_dict(torch.load('best_aves_classifier.pt'))
model.eval()

# Test evaluation
with torch.no_grad():
    test_outputs = model(X_test.to(device))
    _, predicted = torch.max(test_outputs.data, 1)
    test_acc = 100 * (predicted == y_test.to(device)).sum().item() / len(y_test)
    
    # Calculate detailed metrics
    y_pred = predicted.cpu().numpy()
    y_true = y_test.numpy()
    
    # Print results
    print("\nTest Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(conf_matrix, index=categories, columns=categories)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df)
    
    # Save results
    results_dir = 'results/aves_classifier'
    os.makedirs(results_dir, exist_ok=True)
    cm_df.to_csv(f'{results_dir}/confusion_matrix.csv')
    report_df.to_csv(f'{results_dir}/classification_report.csv') 