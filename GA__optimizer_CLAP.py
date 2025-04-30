import numpy as np
import random
import json
from torch import load
from transformers import ClapModel
from model.utils import train_audio_model
from msclap import CLAP


class GA():
    def __init__(self, HYPERPARAM_RANGES, data_path, file_path, num_classes=4):
        self.HYPERPARAM_RANGES = HYPERPARAM_RANGES
        self.data_path = data_path
        self.file_path = file_path
        #self.processor = processor
        self.num_classes = num_classes

    def objective_function(self, individual, exp_name):
        """
        Evalúa el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
        """
        # load the encoder model
        prompt = "This is a sound of"
        # define the number of batch_size
        batch_size = individual['batch_size']
        k_shots = individual['k_shots']
        num_epochs = individual['num_epochs']
        lr = individual['lr']
        momentum = individual['momentum']
        weight_decay = individual['weight_decay']
        temperature = individual['temperature']
        temperature_closs = individual['temperature_closs']
        alpha = individual['alpha']

        model_train = train_audio_model( self.data_path, self.file_path, self.num_classes,
                                        prompt, k_shots, batch_size, num_epochs, lr, momentum, weight_decay,
                                        temperature,temperature_closs, alpha)
        path_best_model, best_f1score = model_train.train(exp_name)

        return path_best_model, best_f1score

    def test(self, individual, loaded_model):
        """
        Prueba el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
        """
        # load the encoder model
        prompt = "This is a sound of"
        # define the number of batch_size
        batch_size = individual['batch_size']
        k_shots = individual['k_shots']
        num_epochs = individual['num_epochs']
        momentum = individual['momentum']
        weight_decay = individual['weight_decay']
        lr = individual['lr']
        temperature = individual['temperature']
        temperature_closs = individual['temperature_closs']
        alpha = individual['alpha']

        model_train = train_audio_model( self.data_path, self.file_path, self.num_classes,
                                        prompt, k_shots, batch_size, num_epochs, lr, momentum, weight_decay,
                                        temperature,temperature_closs, alpha)
        metrics = model_train.test(loaded_model)

    def is_diverse(self, new_individual, population):
        """
        Checks if the new individual is sufficiently different from the existing population.
        """
        for individual in population:
            if all(abs(new_individual[key] - individual[key]) < 1e-3 for key in new_individual.keys()):
                return False
        return True

    def initialize_population(self, size):
        population = []
        while len(population) < size:
            individual = {key: random.uniform(*value) if isinstance(value, tuple) else random.choice(value)
                          for key, value in self.HYPERPARAM_RANGES.items()}
            individual['num_epochs'] = int(individual['num_epochs'])
            individual['batch_size'] = int(individual['batch_size'])
            individual['k_shots'] = int(individual['k_shots'])

            if self.is_diverse(individual, population):
                population.append(individual)
        return population

    def combine_solutions(self, parent1, parent2):
        child = {}
        for key in parent1:
            if np.random.rand() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate_solution(self, individual, mutation_rate=0.1):
        mutated = individual.copy()
        for key in self.HYPERPARAM_RANGES:
            if random.random() < mutation_rate:
                range_min, range_max = self.HYPERPARAM_RANGES[key]
                mutation = random.uniform(-0.1, 0.1) * (range_max - range_min)
                mutated[key] = max(range_min, min(range_max, mutated[key] + mutation))
                if key in ['num_epochs', 'batch_size', 'k_shots']:
                    mutated[key] = int(mutated[key])
        return mutated

    def tournament_selection(self, population, k=3):
        """
        Tournament selection to choose a parent.
        """
        players_id = np.random.choice(range(len(population)), k, replace=False)
        players = [population[i] for i in players_id]
        sorted_players = sorted(players, key=lambda x: x[1], reverse=True)
        winner = sorted_players[0][0]  # Return the solution of the best player
        return winner

    def GA(self, population_size=10, generations=20, elite_size=5, mutation_rate=0.1):
        # Generate the initial population
        population = self.initialize_population(population_size)
        best_ind_gen = {}
        global_best_individual = None
        global_best_score = -float("inf")
        global_model_params_path = None

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            # Evaluate the population
            population_scores = []
            counter = 0
            for individual in population:
                counter += 1
                exp_name = f"Best/{generation}_{counter}"
                path_best_model, best_f1score = self.objective_function(individual, exp_name)
                population_scores.append((individual, [path_best_model, best_f1score]))
                if best_f1score > global_best_score:
                    global_best_score = best_f1score
                    global_best_individual = individual
                    global_model_params_path = path_best_model
            population_scores.sort(key=lambda x: x[1][1], reverse=True)


            # Keep the elite
            elite = [ind[0] for ind in population_scores[:elite_size]]
            # Guardar el mejor individuo de la generación actual
            best_individual_generation = population_scores[0]
            best_ind_gen[generation]=best_individual_generation#.append(best_individual_generation)

            # Generate new individuals
            new_population = elite.copy()
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population_scores, k=3)
                parent2 = self.tournament_selection(population_scores, k=3)
                child = self.combine_solutions(parent1, parent2)
                child = self.mutate_solution(child, mutation_rate)
                if self.is_diverse(child, new_population):
                    new_population.append(child)

            population = new_population

        # Al final del proceso, el mejor global es:
        print(f"Global best individual: {global_best_individual}")
        print(f"Global best score: {global_best_score}")

        with open('best_ind_gen.json', 'w') as f:
            json.dump(best_ind_gen, f, indent=4)

        # Test the model with the best hyperparameters
        loaded_model = load(global_model_params_path)
        self.test(global_best_individual, loaded_model)
        return global_best_individual, global_best_score


HYPERPARAM_RANGES = {
    'num_epochs': (1, 100),       # Número de épocas
    'batch_size': (2, 64),        # Tamaño del batch
    'lr': (0.05, 0.1),            # Tasa de aprendizaje
    'k_shots': (1, 2),            # Número de "k-shots" en few-shot learning
    'momentum': (0.8, 0.99),      # Momentum para SGD
    'weight_decay': (1e-5, 1e-2),  # Weight decay (L2 regularización)
    'temperature':  (0.1,6),
    'temperature_closs':  (0.001,1),
    'alpha': (0.2,0.8)
}

# The path with all the data (train, test, val)
data_path = '/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'

# Verify if the features are extracted yet
file_path = 'CLAP_audio_text_embeddings.pt'#'Biolingual_audio_text_embeddings.pt'

model = ClapModel.from_pretrained("davidrrobinson/BioLingual")

path='/home/julian/PycharmProjects/pythonProject/datos/weights_clap/CLAP_weights_2023.pth'
clap_model = CLAP(path,version = '2023', use_cuda=True)
num_classes=4
# Llamamos a Scatter Search
HPO_model=GA(HYPERPARAM_RANGES,data_path,file_path,num_classes)
best_hyperparams, best_score = HPO_model.GA( population_size=20,  generations=10,elite_size=5,mutation_rate=0.2)




print("Mejores hiperparámetros encontrados:")
print(best_hyperparams)