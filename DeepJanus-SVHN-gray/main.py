import random
import numpy as np
from deap import base, creator, tools
from deap.tools.emo import selNSGA2
import h5py

import vectorization_tools
from mnist_member import MnistMember
from digit_mutator import DigitMutator

from predictor import Predictor
from timer import Timer
from utils import print_archive, print_archive_experiment, set_all_seeds
import archive_manager
from individual import Individual
from scipy.io import loadmat
from config import NGEN, \
    POPSIZE, INITIALPOP, \
    RESEEDUPPERBOUND, GENERATE_ONE_ONLY, \
    STOP_CONDITION, STEPSIZE, DJ_DEBUG, EXPLABEL, \
    IMG_SIZE, IMG_CHN, SEED, confidence_100

set_all_seeds(seed=SEED)
# Original DJ method
# Load the dataset.
# hf = h5py.File(DATASET, 'r')
# x_test = hf.get('xn')
# x_test = np.array(x_test)
# y_test = hf.get('yn')
# y_test = np.array(y_test)

#
import os
import cv2

def load_icse_data(confidence_is_100, label):
    if confidence_is_100:
        dataset_path = os.path.join("original_dataset/final-svhn-ices-10/svhn/100/", str(label))
    else:
        dataset_path = os.path.join("original_dataset/final-svhn-ices-10/svhn/not_100/", str(label))
    # List all files in the directory
    file_list = os.listdir(dataset_path)

    image_files = [f for f in file_list if f.endswith('.png')]

    # Initialize an empty list to store the images
    images = []

    # Load each image and append to the list
    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        # Load image in grayscale mode
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    # Convert list of images to a single 3D numpy array (matrix)
    x_test = np.array(images)

    print(x_test.shape)
    y_test = np.array([label] * len(x_test))
    return x_test, y_test

x_test, y_test = load_icse_data(confidence_100, EXPLABEL)

#datasetLoc = '/home/vin/PycharmProjects/dola/DistributionAwareDNNTesting/SVHN_dx/dataset/test_32x32.mat'
#test_data = loadmat(datasetLoc)
#x_test = np.array(test_data['X'])
# Normalize data.
#x_test = np.moveaxis(x_test, -1, 0)

#y_test = test_data['y']
#y_test[y_test == 10] = 0
#y_test = np.array(y_test)[:,]

#y_test = y_test[np.where(y_test==EXPLABEL)]
#x_test = x_test[np.where(y_test==EXPLABEL)]
## TODO: check labels


# Fetch the starting seeds from file
starting_seeds = [i for i in range(len(y_test))]
random.shuffle(starting_seeds)
all_seeds = starting_seeds
starting_seeds = starting_seeds[:POPSIZE]

# for seed in starting_seeds:
#     from matplotlib import pyplot as plt
#     seed_image = x_test[seed]
#     plt.imsave("seed"+str(seed)+".png", seed_image, cmap='gray')

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
# Define the individual.
creator.create("Individual", Individual, fitness=creator.FitnessMulti)


def generate_digit(seed):
    seed_image = x_test[int(seed)]
    label = y_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    return MnistMember(xml_desc, label, seed)


def ind_from_seed(seed):
    Individual.COUNT += 1
    #print(Individual.COUNT)
    if not GENERATE_ONE_ONLY:
        digit1, digit2, distance_inputs = \
            DigitMutator(generate_digit(seed)).generate()
    else:
        digit1 = generate_digit(seed)
        digit2 = digit1.clone()
        distance_inputs = DigitMutator(digit2).mutate()

    individual = creator.Individual(digit1, digit2)
    individual.members_distance = distance_inputs
    individual.seed = seed
    return individual


def generate_individual():
    if INITIALPOP == 'seeded':
        # Choose sequentially the inputs from the seed list.
        # NOTE: number of seeds should be no less than the initial population
        assert (len(starting_seeds) == POPSIZE)
        seed = starting_seeds[Individual.COUNT]
        Individual.SEEDS.add(seed)
    # elif INITIALPOP == 'random':
    else:
        # Choose randomly a file in the original dataset.
        seed = random.choice(starting_seeds)
        Individual.SEEDS.add(seed)
    individual = ind_from_seed(seed)
    return individual


def reseed_individual(seeds):
    # Chooses randomly the seed among the ones that are not covered by the archive
    if len(starting_seeds) > len(seeds):
        seed = random.sample(set(starting_seeds) - seeds, 1)[0]
    else:
        #seed = random.choice(starting_seeds)
        seed = random.choice(all_seeds)
    individual = ind_from_seed(seed)
    return individual


# Evaluate an individual.
def evaluate_individual(individual, current_solution):
    individual.evaluate(current_solution)
    return individual.aggregate_ff, individual.misclass


def mutate_individual(individual):
    Individual.COUNT += 1
    # Select one of the two members of the individual.
    if random.getrandbits(1):
        distance_inputs = DigitMutator(individual.m1).mutate(reference=individual.m2)
    else:
        distance_inputs = DigitMutator(individual.m2).mutate(reference=individual.m1)
    individual.reset()
    individual.members_distance = distance_inputs


toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", selNSGA2)
toolbox.register("mutate", mutate_individual)


def pre_evaluate_batch(invalid_ind):
    batch_members = [i.m1
                     for i in invalid_ind
                     if i.m1.predicted_label is None]
    batch_members += [i.m2
                      for i in invalid_ind
                      if i.m2.predicted_label is None]

    batch_img = [m.purified for m in batch_members]
    batch_img = np.reshape(batch_img, (-1, IMG_SIZE, IMG_SIZE, IMG_CHN))

    batch_label = ([m.expected_label for m in batch_members])

    predictions, confidences = (Predictor.predict(img=batch_img,
                                                  label=batch_label))

    for member, prediction, confidence \
            in zip(batch_members, predictions, confidences):
        member.confidence = confidence
        member.predicted_label = prediction
        if member.expected_label == member.predicted_label:
            member.correctly_classified = True
        else:
            member.correctly_classified = False


def main(rand_seed=SEED):
    random.seed(rand_seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # Generate initial population.
    print("### Initializing population ....")
    population = toolbox.population(n=POPSIZE)
    # TODO: test digits
    # for ind in population:
    #    digit = ind.m1
    #    digit.export(ind.seed)
    #exit()

    # Evaluate the individuals with an invalid fitness.
    # Note: the fitnesses are all invalid before the first iteration since they have not been evaluated
    invalid_ind = [ind for ind in population]
    to_evaluate_ind = [ind for ind in population if ind.misclass is None]

    pre_evaluate_batch(to_evaluate_ind)

    # Note: the sparseness is calculated wrt the archive. It can be calculated wrt population+archive
    # Therefore, we pass to the evaluation method the current archive.
    fitnesses = [toolbox.evaluate(i, archive.get_archive()) for i in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update archive with the individuals on the decision boundary.
    for ind in population:
        if ind.archive_candidate:
            archive.update_archive(ind)

    print("### Number of Individuals generated in the initial population: " + str(Individual.COUNT))

    # This is just to assign the crowding distance to the individuals (no actual selection is done).
    population = toolbox.select(population, len(population))

    record = stats.compile(population)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    condition = True
    gen = 1
    while condition:
        # Vary the population.
        offspring = tools.selTournamentDCD(population, 8) #len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Reseeding
        if len(archive.get_archive()) > 0:
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds

            for i in range(seed_range):
                population[len(population) - i - 1] = reseed_individual(candidate_seeds)

            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    population[i] = reseed_individual(candidate_seeds)

        # Mutation.
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals
        # NOTE: all individuals in both population and offspring are evaluated to assign crowding distance.
        invalid_ind = [ind for ind in population + offspring]
        pre_evaluate_batch(invalid_ind)

        fitnesses = [toolbox.evaluate(i, archive.get_archive()) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for ind in population + offspring:
            if ind.archive_candidate:
                archive.update_archive(ind)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        if DJ_DEBUG and gen % STEPSIZE == 0:
            archive.create_report(x_test, Individual.SEEDS, gen)

        # Update the statistics with the new population
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        gen += 1

        if STOP_CONDITION == "iter":
            if gen == NGEN:
                condition = False
        elif STOP_CONDITION == "time":
            if not Timer.has_budget():
                condition = False

    archive.create_report(x_test, Individual.SEEDS, gen)
    print(logbook.stream)

    return population


if __name__ == "__main__":
    archive = archive_manager.Archive()
    pop = main()
    print_archive_experiment(archive.get_archive())

    print_archive(archive.get_archive())
    #archive.create_report(x_test, Individual.SEEDS, 'final')
    print("GAME OVER")
