
from config import REPORT_NAME

import numpy as np
import matplotlib.pyplot as plt
import csv
from os.path import join

dst1 = join("./runs/"+"run_1698321150", REPORT_NAME)
dst2 = join("./runs/"+"run_1698324099", REPORT_NAME)
class DataPreprocess:
    def __init__(self, dist):
        self.path = dist
        self.mutation_iterations = list()
        self.number_per_iteration = list()
        self.population_size = 0
        self.iteration = 0
        self.misclass_number = 0
        self.read_data()
        self.preprocess()
    def read_data(self):
        with open(self.path, mode='r') as report_file:
            csvreader = csv.reader(report_file)

            for rowid, row in enumerate(csvreader):
                if rowid == 1:
                    print(row)
                    [population_size, iteration, misclass_number, self.mutation_type, _] = row
                    self.population_size = int(population_size)
                    self.iteration = int(iteration)
                    self.misclass_number = int(misclass_number)
                if rowid >= 4:
                    if row[3] == "True":
                        self.mutation_iterations.append(int(row[5]))
                    else:
                        self.mutation_iterations.append(iteration)

    def preprocess(self):
        for i in range(self.iteration):
            number = self.mutation_iterations.count(i)
            if i == 0:
                self.number_per_iteration.append(number)
            else:
                self.number_per_iteration.append(self.number_per_iteration[-1] + number)
        assert (self.number_per_iteration[-1] == self.misclass_number)


data_set1 = DataPreprocess(dst1)
data_set2 = DataPreprocess(dst2)

def hist_plot(mutation_iterations, iteration):
    bins = 20
    plt.hist(mutation_iterations, bins=bins)#, color='skyblue', edgecolor='black')
    #plt.xticks(np.linspace(0, iteration, bins))
    plt.xlabel('Mutation iterations')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mutation iteration')



def line_plot(number_per_iteration):
    plt.plot(number_per_iteration)
    plt.xlabel('Mutation iterations')
    plt.ylabel('Number of misbehaviours')
    plt.title('Histogram of Mutation iteration')




line_plot(data_set1.number_per_iteration)
line_plot(data_set2.number_per_iteration)
plt.legend([data_set1.mutation_type, data_set2.mutation_type])
plt.show()


hist_plot(data_set1.mutation_iterations, data_set1.iteration)
hist_plot(data_set2.mutation_iterations, data_set2.iteration)
plt.legend([data_set1.mutation_type, data_set2.mutation_type])
plt.show()
