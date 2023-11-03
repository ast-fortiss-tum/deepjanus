import random
import mutation_manager
import rasterization_tools
import vectorization_tools
from config import MUTOPPROB, MUTATION_TYPE, MUTEXTENT
from utils import get_distance


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        self.control_points = list()

    def mutate(self):
        # TODO: be more specific on mutation types (do not use numbers)
        # Select mutation operator.
        if MUTATION_TYPE == "random":
            rand_mutation_probability = random.uniform(0, 1)
            if rand_mutation_probability >= MUTOPPROB:
                mutation = 1
            else:
                mutation = 2
        elif MUTATION_TYPE == "attention-based":
            mutation = 3

        condition = True
        counter_mutations = 0
        distance_inputs = 0

        while condition:
            counter_mutations += 1
            #mutant_vector = mutation_manager.mutate(self.digit.xml_desc, mutation, counter_mutations/20)
            mutant_vector = mutation_manager.mutate(self.digit.purified, self.digit.xml_desc, self.digit.attention, mutation, MUTEXTENT)#counter_mutations / 20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.digit.purified, rasterized_digit)
            if distance_inputs != 0:
                condition = False

        self.digit.reset()
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit



