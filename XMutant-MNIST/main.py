import numpy as np
import random
import csv

from config import DATASET, POPSIZE, STOP_CONDITION, NGEN
from population import Population
from timer import Timer


def main():

    pop = Population()
    print(f" Population size {pop.size}")


    # Collect data
    field_names = ["id", "misclass ", "predicted label"]
    data = []

    condition = True
    gen = 1

    while condition:

        pop.evaluate_population(gen)
        confidences = [ind.confidence for ind in pop.population_to_mutate]
        if len(confidences) > 0:
            print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
                  .format(*[gen, pop.misclass_number, len(confidences), np.mean(confidences),
                            np.min(confidences), np.max(confidences)]))
            pop.mutate()
            gen += 1
            if STOP_CONDITION == "iter":
                if gen == NGEN:
                    condition = False
            elif STOP_CONDITION == "time":
                if not Timer.has_budget():
                    condition = False
        else:
            print("All mutations finished, early termination")

    pop.evaluate_population(gen)
    confidences = [ind.confidence for ind in pop.population_to_mutate]
    print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
          .format(*[gen, pop.misclass_number, len(confidences), np.mean(confidences),
                    np.min(confidences), np.max(confidences), ]))

    print("MUTATION FINISHED")
    # record data
    pop.create_report(gen)
    print("REPORT GENERATED")

if __name__ == "__main__":
    main()
