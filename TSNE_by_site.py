import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
#sites Kale
point={'SMA03286':'KLSA01','SMA03294':'KLSA02','SMA03332':'KLSA03',
       'SMA03322':'KLSA04','SMA03210':'KLSA05','SMA03328':'KLSA06',
       'SMA03297':'KLSA07','SMA03175ICP':'KLSA08','SMA03247':'KLSA09',
       'SMA03327':'KLSA10','SMA03393':'KLSA11','SMA03406':'KLSA12',
       'SMA03320':'KLSA13','SMA03126':'KLSA14','SMA03326':'KLSA15',
       'SMA03330':'KLSA16','SMA03251':'KLSA17','SMA03411':'KLSA18'}
# Load the embeddings
data = torch.load('Biolingual_audio_text_embeddings.pt')
# text_embeddings = data['text_embeddings']
#data = torch.load('AVES_audio_embeddings.pt')
audio_embeddings = data['audio_embeddings']

# Prepare data for t-SNE
embeddings = []
labels = []
species_site = []  # To label by species and site

# Extract audio embeddings with species and site labels
for _, audios in audio_embeddings.items():
    for name_audio, (category, embedding) in audios.items():
        if category == 'Leptodactylus fuscus':
            parts = name_audio.split('_')
            species = parts[0]       # Extract species
            site = parts[1]           # Extract site
            if point[site] == 'KLSA17':
                embeddings.append(embedding.cpu().numpy().flatten())  # Flatten if necessary
                labels.append(category)
                species_site.append(f"{category}_{point[site]}")  # Combine species and site

# Convert lists to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)
species_site = np.array(species_site)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Plot t-SNE results
plt.figure(figsize=(12, 10))

# Plot each category with different markers for species and site
unique_labels = set(species_site)
for label in unique_labels:
    indices = (species_site == label)
    if np.any(indices):
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=label,
            alpha=0.7,
            marker='o'
        )

# Group legend by removing duplicate labels
handles, _ = plt.gca().get_legend_handles_labels()
by_label = dict(zip(_, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

plt.title("t-SNE Visualization of Audio Embeddings by Species and Site")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
