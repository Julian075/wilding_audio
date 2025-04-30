from torch.utils.data import Dataset,Subset
import os
import numpy as np
import soundfile as sf
import torch
import random
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import model.audio_model as md
from torch.utils.data import DataLoader




def seed_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

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


def biolingual_features_audio(audio_path,model,processor,max_length,device):
    # Load the audio file
    audio_sample, sample_rate = load_aud(audio_path,max_length)

    with torch.no_grad():
        inputs = processor(audios=audio_sample,sample_rate=sample_rate, return_tensors="pt").to(device)
        audio_embed = model.get_audio_features(**inputs)
    return audio_embed

def biolingual_features_text(prompt,model,processor,device):
    with torch.no_grad():
        text_inputs = processor(text=prompt ,return_tensors="pt").to(device)
        text_embed = model.get_text_features(**text_inputs)
    return text_embed


def extract_features(prompt, data_path, num_classes, model, processor, max_length, device,
                     max_samples_per_category=None):
    audio_embeddings = {'train': {}, 'val': {}, 'test': {}}
    text_embeddings = {}
    id = 0

    for folder_type in os.listdir(data_path):
        for category in os.listdir(os.path.join(data_path, folder_type)):
            if category not in text_embeddings:
                text_embeddings[category] = {
                    'embeddings': biolingual_features_text(f'{prompt} {category}', model, processor, device),
                    'id': id
                }
                id += 1

            audio_files = os.listdir(os.path.join(data_path, folder_type, category))

            # Limitar la cantidad de audios en `train`
            if folder_type == 'train' and max_samples_per_category is not None:
                random.shuffle(audio_files)
                audio_files = audio_files[:max_samples_per_category]

            for name_audio in audio_files:
                audio_path = os.path.join(data_path, folder_type, category, name_audio)
                target = text_embeddings[category]['id']
                one_hot_target = np.zeros(num_classes)
                one_hot_target[target] = 1
                audio_embeddings[folder_type][name_audio] = (
                    one_hot_target,
                    biolingual_features_audio(audio_path, model, processor, max_length, device)
                )

    # Guardar embeddings en un archivo
    save_path = 'Biolingual_audio_text_embeddings.pt'
    torch.save({'audio_embeddings': audio_embeddings, 'text_embeddings': text_embeddings}, save_path)
    print(f"Embeddings saved to {save_path}")


class AudioTextEmbeddingDataset(Dataset):
    def __init__(self, audio_embeddings):
        self.data = list(audio_embeddings.items())  # [(nombre_audio, (one_hot_target, audio_embedding))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extraer el nombre del archivo, la etiqueta y el embedding de audio
        name_audio, (one_hot_target, audio_embedding) = self.data[idx]

        return name_audio, audio_embedding, one_hot_target

def load_datasets_from_embeddings(embedding_file_path):

    data = torch.load(embedding_file_path)
    audio_embeddings = data['audio_embeddings']
    text_embeddings = data['text_embeddings']

    datasets = {}
    for folder_type, embeddings in audio_embeddings.items():
        datasets[folder_type] = AudioTextEmbeddingDataset(embeddings)

    return datasets,text_embeddings






class train_audio_model:
    def __init__(self,data_path,file_path,num_classes,k_shots,batch_size,num_epochs,lr, momentum, weight_decay,temperature,temperature_closs,alpha):
        # seed
        seed_seed(42)
        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model=model.to(self.device) # ClapModel.from_pretrained("davidrrobinson/BioLingual")
        #self.processor=processor
        # enter the max length in s of the input audios
        self.max_length = 3
        self.data_path=data_path
        # Verify if the features are extracted yet
        self.file_path=file_path
        self.num_classes = num_classes
        #
        self.num_epochs=num_epochs
        self.lr=lr
        self.momentum= momentum
        self.weight_decay = weight_decay
        self.alpha=float(alpha)
        self.temperature=float(temperature)
        self.temperature_closs=temperature_closs
        #
        #if not os.path.exists(file_path):
        #    prompt='A sound of'
        #    extract_features(prompt, data_path, num_classes, model, processor, self.max_length, self.device)

        self.datasets, self.text_embeddings = load_datasets_from_embeddings(file_path)

        self.k_shots=k_shots
        self.batch_size=batch_size

        # config text_embeddings
        self.categories = {}
        self.matrix_text_embeddings = []
        for class_name in self.text_embeddings.keys():
            self.categories[class_name] = self.text_embeddings[class_name]['id']
            self.matrix_text_embeddings.append(torch.tensor(self.text_embeddings[class_name]['embeddings']))

        self.matrix_text_embeddings = (torch.stack(self.matrix_text_embeddings).squeeze()).to(self.device)
        # Create a DataLoader
        self.dataloader = {}
        for name, dataset in self.datasets.items():
            if name == 'train':
                self.dataloader[name] = DataLoader(dataset, batch_size=1, shuffle=True)
                # Usar la función para dividir el dataset
                self.support_vectors, self.query_vectors = self.select_k_shots_embeddings(self.dataloader[name], self.k_shots)
                self.query_vectors = DataLoader(self.query_vectors, batch_size=batch_size, shuffle=True)
                self.dataloader[name] = {'support_vectors': self.support_vectors, 'query_vectors': self.query_vectors}
            else:
                self.dataloader[name] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def select_k_shots_embeddings(self,dataset, k_shots):
        # Initialize dictionaries to hold the support and query sets
        support_data = []
        query_data = {}

        # Group the embeddings and labels by category using the position of `1` in one_hot_target
        support_embeddings = {}
        query_embeddings = {}
        selected_idx = []

        for idx, (name_audio, audio_embedding, one_hot_target) in enumerate(dataset):
            #if not audio_embedding.requires_grad:
            #    audio_embedding = audio_embedding.detach().clone().requires_grad_(True)
            #if not one_hot_target.requires_grad:
            #    one_hot_target = one_hot_target.detach().clone().requires_grad_(True)

            target_class = one_hot_target.argmax().item()  # Extract the class from the one-hot encoding

            # Initialize the dictionary entries if they don't exist
            if target_class not in support_embeddings:
                support_embeddings[target_class] = []

            if len(support_embeddings[target_class]) < k_shots:
                support_embeddings[target_class].append((audio_embedding, one_hot_target))
                support_data.append((name_audio, (audio_embedding, one_hot_target)))
                selected_idx.append(idx)
            else:
                if target_class not in query_embeddings:
                    query_embeddings[target_class] = []

                query_embeddings[target_class].append((audio_embedding, one_hot_target))
                query_data[name_audio] = (one_hot_target, audio_embedding)

        # Create the datasets for support and query sets
        query_dataset = AudioTextEmbeddingDataset(query_data)

        return support_data, query_dataset

    def train(self,path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # training
        patience=5
        audio_model = md.Audio_classifier(nclasses=self.num_classes, k_shots=self.k_shots, embed_size=512,  support_audio=self.support_vectors)
        audio_model.to(self.device)

        optimizer = torch.optim.SGD(audio_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss()  # loss function
        path_best_model = os.path.join(path, "audio_classifier_model.pt")



        best_f1score = 0.0
        best_val = 0.0
        train_loader = self.query_vectors

        for epoch in range(self.num_epochs):
            audio_model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            prompts_test = self.matrix_text_embeddings.squeeze()
            for batch in train_loader:
                # batch = name_audio, audio_embedding, one_hot_target
                audio_emb_test = (batch[1].squeeze()).to(self.device)
                one_hot_vectors = (batch[2].squeeze()).to(self.device)


                # Forward pass

                outputs = audio_model(audio_emb_test, prompts_test, self.temperature, self.alpha)

                # Calcular pérdida
                if len(one_hot_vectors.shape) == 1:
                    labels = one_hot_vectors.unsqueeze(0).argmax(dim=1)
                else:
                    labels = one_hot_vectors.argmax(dim=1)
                loss = audio_model.Contrastive_loss(outputs, labels, self.temperature_closs)#criterion(outputs, labels)


                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()

                # Precisión
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Imprimir la pérdida promedio de cada época
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

            # Validación
            audio_model.eval()
            val_loss = 0.0
            all_labels = []
            all_predictions = []
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in self.dataloader['val']:


                    audio_emb_test = batch[1].squeeze().to(self.device)
                    if len(batch[2].shape)<3:
                        one_hot_vectors = batch[2].to(self.device)
                    else:
                        one_hot_vectors = batch[2].squeeze().to(self.device)
                    prompts_test = self.matrix_text_embeddings.squeeze().to(self.device)

                    # Forward pass
                    outputs = audio_model(audio_emb_test, prompts_test,self.temperature,self.alpha)

                    # Calcular pérdida
                    labels = one_hot_vectors.argmax(dim=1)
                    loss = audio_model.Contrastive_loss(outputs, labels, self.temperature_closs)#criterion(outputs, labels)
                    val_loss += loss.item()

                    # Precisión
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            # Pérdida promedio y precisión de validación
            avg_val_loss = val_loss / len(self.dataloader['val'])
            accuracy = 100 * correct / total
            f1 = f1_score(all_labels, all_predictions, average="macro")
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}],  Val Loss: {avg_val_loss:.4f}, Val F1 Score: {f1:.4f}, Val Accuracy: {accuracy:.2f}%")
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

            # Save the model if F1 score improves
            #if f1 > best_f1score:
            #    best_f1score = f1
            if best_val < accuracy:
                best_val = accuracy
                torch.save(audio_model, path_best_model)
                patience=5
                print(f"Save model with F1 Score: {f1:.4f}")
            else:
                patience=patience-1
            if patience<=0:
                print(f"The model doesn't improve in 5 epochs")
                break
            # if  best_val < accuracy:
            #    best_val = accuracy
            #    # Guardar el modelo completo (incluyendo los pesos y la arquitectura)
            #    torch.save(model, path_best_model)
            #    print(f"Save model")

        print("Complete training.")
        return path_best_model, best_val


    def test(self,loaded_model):
        # Evaluación en el conjunto de prueba
        loaded_model.to(self.device)
        loaded_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        test_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.dataloader['test']:
                # batch = (name_audio, audio_embedding, one_hot_target)
                audio_emb_test = batch[1].squeeze().to(self.device)
                if len(batch[2].shape) < 3:
                    one_hot_vectors = batch[2].to(self.device)
                else:
                    one_hot_vectors = batch[2].squeeze().to(self.device)
                prompts_test = self.matrix_text_embeddings.squeeze().to(self.device)

                # Forward pass
                outputs = loaded_model(audio_emb_test, prompts_test, self.temperature,self.alpha)

                # Calcular pérdida
                labels = one_hot_vectors.argmax(dim=1)
                loss = loaded_model.Contrastive_loss(outputs, labels, self.temperature_closs)#criterion(outputs, labels)
                test_loss += loss.item()

                # Precisión y predicciones
                _, predicted = torch.max(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Pérdida promedio
        avg_test_loss = test_loss / len(self.dataloader['test'])

        # Generar métricas
        report = classification_report(all_labels, all_predictions, target_names=self.text_embeddings.keys(), output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        f1_macro = report['macro avg']['f1-score']
        # Imprimir métricas
        print(f"Test Loss: {avg_test_loss:.4f}")
        print("\nClassification Report:\n", report)
        print("\nConfusion Matrix:\n", conf_matrix)

        # Retornar métricas como un diccionario
        return {
            "test_loss": avg_test_loss,
            "classification_report": report,
            "f1_macro": f1_macro,
            "confusion_matrix": conf_matrix
        }

    def train_cross_validation(self, exp_name):
        """
        Realiza validación cruzada utilizando los folds preprocesados.
        """
        cv_metrics = []
        for fold_idx in range(4):  # 4 folds
            print(f"\n===== Starting Fold {fold_idx + 1} =====\n")
            fold_name = f'fold_{fold_idx}'

            # Configurar los conjuntos de entrenamiento y validación
            self.dataloader['train'] = self.datasets[fold_name]['train']
            self.dataloader['val'] = self.datasets[fold_name]['val']

            # Entrenar y evaluar en este fold
            path_best_model, _ = self.train(exp_name + f"/fold_{fold_idx}")
            loaded_model = torch.load(path_best_model)
            metrics = self.test(loaded_model)
            cv_metrics.append(metrics)

        return cv_metrics








