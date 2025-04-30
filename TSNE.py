import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the embeddings
data = torch.load('Biolingual_audio_text_embeddings.pt')
#audio_embeddings = data['audio_embeddings']
text_embeddings = data['text_embeddings']
data = torch.load('AVES_audio_embeddings.pt')
audio_embeddings = data['audio_embeddings']

# Prepare data for t-SNE
embeddings = []
labels = []
data_type = []  # To distinguish between test and train/val

# Extract audio embeddings with combined labels for each category and data type
for folder_type in audio_embeddings.keys():
    for name_audio, (category, embedding) in audio_embeddings[folder_type].items():
        embeddings.append(embedding.cpu().numpy().flatten())  # Flatten if necessary
        labels.append(f'{category}_audio')
        data_type.append(folder_type)  # Mark as train, val, or test

## Extract text embeddings with combined labels for each category and data type
#for category, embedding in text_embeddings.items():
#    embeddings.append(embedding.cpu().numpy().flatten())  # Flatten if necessary
#    labels.append(f'{category}_text')
#    data_type.append('train')  # Mark text embeddings as train for grouping purposes

# Convert lists to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)
data_type = np.array(data_type)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Plot t-SNE results
plt.figure(figsize=(12, 10))

# Plot each category with different markers for audio and text
unique_labels = set(labels)
for label in unique_labels:
    for dtype in ['train', 'val', 'test']:
        indices = (labels == label) & (data_type == dtype)
        if np.any(indices):
            marker = 'o' if 'audio' in label else '^'  # Use circles for audio and triangles for text
            edgecolor = 'black' if dtype == 'test' else 'none'  # Add black edge for test data
            plt.scatter(
                tsne_results[indices, 0],
                tsne_results[indices, 1],
                label=f'{label}_{dtype}',
                alpha=0.7,
                marker=marker,
                edgecolor=edgecolor
            )

# Group legend by removing duplicate labels
handles, _ = plt.gca().get_legend_handles_labels()
by_label = dict(zip(_, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("t-SNE Visualization Audio embeddings AVES Model")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
