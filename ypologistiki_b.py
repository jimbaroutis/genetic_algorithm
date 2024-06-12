import numpy as np
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deap import base, creator, tools
import matplotlib.pyplot as plt

# Δομή δεδομένων και φόρτωση
dtype = {
    'names': ('id', 'text', 'metadata', 'region_main_id', 'region_main', 'region_sub_id', 'region_sub', 'date_str', 'date_min', 'date_max', 'date_circa'),
    'formats': ('i4', 'U500', 'U500', 'i4', 'U100', 'i4', 'U100', 'U50', 'f8', 'f8', 'f8')
}

dataset = np.loadtxt('iphi2802.csv', delimiter='\t', dtype=dtype, skiprows=1, encoding='utf-8')

# Φιλτράρισμα των δεδομένων για συγκεκριμένη περιοχή
records = dataset[dataset['region_main_id'] == 1693]
texts = records['text'].tolist()

# Δημιουργία λεξιλογίου
vocabulary_size = 1678
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=vocabulary_size)
tfidf_matrix = vectorizer.fit_transform(texts)
vocabulary = vectorizer.get_feature_names_out()
vocabulary_dict = {word: idx + 1 for idx, word in enumerate(vocabulary)}

# Εισαγωγή της επιγραφής που θα συγκρίνουμε
query_text = "αλεξανδρε ουδις"
texts.append(query_text)
tfidf_matrix = vectorizer.fit_transform(texts)

# Υπολογισμός ομοιότητας συνημιτόνου
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
similarity_df = pd.DataFrame(cosine_similarities.T, columns=['similarity'])
similarity_df['text'] = texts[:-1]

# Top-5 πιο κοντά κείμενα
top_n = 5
top_similar_texts = similarity_df.nlargest(top_n, 'similarity')
top_vectors = vectorizer.transform(top_similar_texts['text'].tolist())

# Ορισμός τύπων για την ακεραια κωδικοποίηση
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual(vocabulary, vocabulary_dict):
    word1, word2 = random.sample(vocabulary, 2)
    return creator.Individual([vocabulary_dict[word1], vocabulary_dict[word2]])

toolbox = base.Toolbox()
toolbox.register("individual", create_individual, vocabulary=list(vocabulary), vocabulary_dict=vocabulary_dict)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_individual(individual, vocabulary_dict):
    reverse_vocabulary_dict = {v: k for k, v in vocabulary_dict.items()}
    words = [reverse_vocabulary_dict[idx] for idx in individual]
    return f" {words[0]} αλεξανδρε ουδις {words[1]}"

def fitness_function(individual):
    record = decode_individual(individual, vocabulary_dict)
    completed_vector = vectorizer.transform([record])
    similarities = cosine_similarity(completed_vector, top_vectors)[0]
    fitness_value = np.sum(similarities)
    return fitness_value,

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop_size = 200
    elite_size = 5
    num_runs = 10  # Ο αριθμός των εκτελέσεων του αλγορίθμου

    best_individuals = []
    num_generations = []
    avg_fitness_per_gen = [] 

    for _ in range(num_runs):
        pop = toolbox.population(n=pop_size)

        # Εξέτασε ολόκληρο τον πληθυσμό
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        CXPB, MUTPB = 0.1, 0.01

        fits = [ind.fitness.values[0] for ind in pop]

        g = 0

        max_generations = 1000
        stagnation_threshold = 50
        improvement_threshold = 0.01
        best_fitness = max(fits)
        stagnation_count = 0
        best_fitnesses = []


        while best_fitness < 100 and g < max_generations:
            g += 1
            print("-- Generation %i --" % g)

            # Επιλογή των ελίτ ατόμων
            elite = tools.selBest(pop, elite_size)

            # Επιλογή των υπόλοιπων ατόμων
            offspring = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Επανένωση των ελίτ ατόμων με τον πληθυσμό
            pop[:] = elite + offspring

            fits = [ind.fitness.values[0] for ind in pop]
            current_best_fitness = max(fits)
            best_fitnesses.append(current_best_fitness)
            print("Best fitness of current generation:", current_best_fitness)

            if current_best_fitness - best_fitness < improvement_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

            best_fitness = current_best_fitness

            if stagnation_count >= stagnation_threshold:
                print("Termination: Stagnation limit reached")
                break

        best_individual = tools.selBest(pop, 1)[0]
        best_individuals.append(best_individual.fitness.values[0])
        num_generations.append(g)
    
    
        # Ενημέρωση των μέσων αποδόσεων για κάθε γενιά
        if len(avg_fitness_per_gen) == 0:
            avg_fitness_per_gen = best_fitnesses
        else:
            avg_fitness_per_gen = [sum(x) / 2 for x in zip(avg_fitness_per_gen, best_fitnesses)]
    
    
    
    avg_best_fitness = np.mean(best_individuals)
    avg_generations = np.mean(num_generations)

    print("-- End of (successful) evolution --")
    print("Average best fitness over all runs:", avg_best_fitness)
    print("Average number of generations:", avg_generations)


# Σχεδίαση της καμπύλης εξέλιξης
    plt.plot(avg_fitness_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Average Best Fitness')
    plt.title('Evolution of Average Best Fitness Over Generations')
    plt.show()

    # Εμφάνιση της καλύτερης επιγραφής
    best_individual = tools.selBest(pop, 1)[0]
    best_record = decode_individual(best_individual, vocabulary_dict)
    print("Καλύτερη επιγραφή:", best_record)


if __name__ == "__main__":
    main()
