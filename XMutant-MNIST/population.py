import h5py
import numpy as np
from os.path import join
import csv

from config import DATASET, POPSIZE, MUTATION_TYPE, REPORT_NAME, MUTEXTENT
from individual import Individual
import vectorization_tools
from attention_manager import AttentionManager
from predictor import Predictor
from digit_mutator import DigitMutator
from folder import Folder

def load_dataset():
    # Load the dataset.
    hf = h5py.File(DATASET, 'r')
    x_test = hf.get('xn')
    x_test = np.array(x_test)
    assert (x_test.shape[0] >= POPSIZE)
    x_test = x_test[0:POPSIZE]
    y_test = hf.get('yn')
    y_test = np.array(y_test)
    y_test = y_test[0:POPSIZE]
    return x_test, y_test


def generate_digit(ind_id, image, label):
    xml_desc = vectorization_tools.vectorize(image)
    return Individual(ind_id, xml_desc, label)


class Population:
    def __init__(self):
        x_test, y_test = load_dataset()
        self.population_to_mutate = [generate_digit(i, image, label) for i, (image, label) in enumerate(zip(x_test, y_test))]
        self.size = len(self.population_to_mutate)
        self.misclass_number = 0
        self.misclass_list = []
        self.mutated_population = []

    def evaluate_population(self, gen_number):
        # batch evaluation for
        #         Individual.predicted_label
        #         Individual.confidence
        #         Individual.misclass
        #         Individual.attention
        #         Population.misclass_number
        # in initialization or after every mutation of all population

        # Prediction
        batch_individual = [ind.purified for ind in self.population_to_mutate]
        batch_individual = np.reshape(batch_individual, (-1, 28, 28, 1))
        batch_label = ([ind.expected_label for ind in self.population_to_mutate])

        if MUTATION_TYPE == "attention-based":
            attmaps = AttentionManager.compute_attention_maps(batch_individual)
        else:
            attmaps = [None] * len(batch_individual)

        predictions, confidences = (Predictor.predict(img=batch_individual,
                                                      label=batch_label))

        # label result and detect misclass
        for ind, prediction, confidence, attmap \
                in zip(self.population_to_mutate, predictions, confidences, attmaps):
            ind.confidence = confidence
            ind.predicted_label = prediction

            ind.misclass = ind.expected_label != ind.predicted_label

            ind.attention = attmap
        for ind in self.population_to_mutate:
            if ind.misclass:
                self.population_to_mutate.remove(ind)
                ind.mutate_attempts = gen_number
                self.mutated_population.append(ind)
                self.misclass_number += 1

    def mutate(self):
        batch_individual = [ind for ind in self.population_to_mutate]
        for ind in batch_individual:
            DigitMutator(ind).mutate()

    def create_report(self,gen_number):
        dst = join(Folder.DST, REPORT_NAME)
        with open(dst, mode='w') as report_file:
            report_writer = csv.writer(report_file,
                                       delimiter=',',
                                       quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            report_writer.writerow(['population size',
                                    'total iteration number',
                                    'misbehaviour number',
                                    'mutation type',
                                    'stride extent'])
            report_writer.writerow([self.size,
                                    gen_number,
                                    self.misclass_number,
                                    MUTATION_TYPE,
                                    MUTEXTENT])

            report_writer.writerow('')

            report_writer.writerow(['id',
                                    'expected_label',
                                    'predicted_label',
                                    'misbehaviour',
                                    'confidence',
                                    'mutate_attempts'])

            for ind in [*self.mutated_population, *self.population_to_mutate]:
                report_writer.writerow([ind.id,
                                        ind.expected_label,
                                        ind.predicted_label,
                                        ind.misclass,
                                        ind.confidence,
                                        ind.mutate_attempts])


if __name__ == "__main__":
    from utils import get_distance
    x_test, y_test = load_dataset()
    POPSIZE = 10
    pop = Population()
    print(f" Population size {pop.size}")
    digit_ini = [ind.purified for ind in pop.population_to_mutate]
    for idx in range(1000):
        digit_cur = [ind.purified for ind in pop.population_to_mutate]

        pop.evaluate_population(idx)
        print([ind.confidence for ind in pop.population_to_mutate])
        pop.mutate()

        #print([get_distance(digit_ini[i], digit_cur[i]) for i in range(POPSIZE)])

    print(f" Misclass number {pop.misclass_number}")

