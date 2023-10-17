import os

import matplotlib

from timer import Timer

matplotlib.use('Agg')

import csv

import copy
from random import seed

from config import *
from utils_one_plus_one import *
from vectorization_tools import *

ADAPTIVE = False
METHOD = METHOD_LIST[0]

# set the seeds for reproducibility
seed(RANDOM_SEED)
np.random.seed(START_SEED)

# Creating CSVs in the MUTANTS_ROOT_FOLDER
run_id = str(Timer.start.strftime('%s'))
DST = MUTANTS_ROOT_FOLDER + DEBUG_OR_VALID + "_ISEED=" + str(START_SEED) + "_NDS=" + str(
    NUMBER_OF_DIGIT_SAMPLES) + "_NM=" + str(NUMBER_OF_MUTATIONS) + "_NR=" + str(
    NUMBER_OF_REPETITIONS) + "_EXT=" + str(EXTENT) + "_NP=" + str(NUMBER_OF_POINTS) + "_SQRS=" + str(
    SQUARE_SIZE) + "_MutType=" + METHOD_LIST[0] + "_ID=" + run_id
makedirs(DST)
csv_path = DST + "/stats.csv"
if os.path.exists(csv_path):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

with open(csv_path, append_write) as f1:
    writer = csv.writer(f1)
    writer.writerow(["IMG_Index", "Algorithm", "Mut_Method", "Label", "Prediction", "Probability", "Iteration"])

csv_path_2 = DST + "/stats_2.csv"
if os.path.exists(csv_path_2):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

with open(csv_path_2, append_write) as f1:
    writer = csv.writer(f1)
    writer.writerow(
        ["IMG_Index", "Label", "Repetition", "Seed", "#Iterations_Att", "#Iterations_Normal", "Winner Method"])

csv_path_3 = DST + "/stats_3.csv"
if os.path.exists(csv_path_3):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

with open(csv_path_3, append_write) as f1:
    writer = csv.writer(f1)
    writer.writerow(["IMG_Index", "Label", "Its_Mean_Att", "Its_Mean_Normal", "Its_Std_Att", "Its_Std_Normal",
                     "#MissClass_found_att", "#MissClass_found_Normal"])

if SAVE_STATS4_CSV == True:
    csv_path_4 = DST + "/stats_4.csv"
    if os.path.exists(csv_path_3):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path_4, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(
            ["Iteration", "Point Mutated Att", "Point Mutated Normal", "List of points to be mutated Att"])

# load MNIST test set and classifier model
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = keras.models.load_model(MODEL)

images, labels, indices_chosen = initializate_list_of_images(x_test, y_test, NUMBER_OF_DIGIT_SAMPLES)

if SHUFFLE_IMAGES:
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

for image_index in range(images.shape[0]):
    print("Mutating image %d/%d" % (image_index + 1, N))
    image = images[image_index].reshape(1, 28, 28)
    label = labels[image_index]

    digit_1 = copy.deepcopy(image)
    digit_2 = copy.deepcopy(image)

    iterations_detection_normal_list = []
    iterations_detection_att_list = []

    for repetition in range(1, NUMBER_OF_REPETITIONS + 1):
        # seed = SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[repetition - 1]
        # print("Seed: ", seed)
        print("Repetition %d/%d" % (repetition, NUMBER_OF_REPETITIONS))

        digit_reshaped_1 = input_reshape_and_normalize_images(digit_1)
        digit_reshaped_2 = input_reshape_and_normalize_images(digit_2)

        iteration_list = []
        fitness_function_att = []
        prediction_function_att = []
        fitness_function_normal = []
        prediction_function_normal = []
        mutant_digit_att_list = []
        mutant_digit_normal_list = []
        xai_images_list = []
        list_of_svg_points_list = []
        pred_input_mutant_att_list = []
        pred_input_mutant_normal_list = []
        mutated_points_att_list = []
        mutated_points_normal_list = []
        list_of_svg_points_list_2 = []
        square_att_coordinates_list = []
        original_svg_points_list = []
        mutated_points_att_list_numeric = []

        ext_att_list = []
        ext_normal_list = []

        miss_classification_found_att = False
        miss_classification_found_normal = False
        method_winner = None
        iterations_detection_att = "NA"
        iterations_detection_normal = "NA"
        iteration = 0
        svg_path_att_mth = None
        svg_path_normal_mth = None
        ext_att = EXTENT_LOWERBOUND
        ext_normal = EXTENT
        number_of_times_fitness_function_does_not_change_att = 0
        number_of_times_fitness_function_does_not_change_normal = 0

        iteration = 0

        while iteration < NUMBER_OF_MUTATIONS:
            # print("Iteration", iteration)
            iteration += 1

            # if a misclassification is found we stop mutating
            if not miss_classification_found_att:
                svg_path_att_mth = get_svg_path(input_reshape_images_reverse(digit_reshaped_1)[0])
                pred_class_mutant_att = model.predict(digit_reshaped_1)
                fitness_mutant_att = evaluate_ff2(pred_class_mutant_att, label)

                mutant_digit_att = digit_reshaped_1
                pred_input_mutant_att = np.argmax(model.predict(mutant_digit_att), axis=-1)

                # Generating Mutant Candidate
            mutant_digit_att_candidate, list_of_svg_points, xai, point_mutated, square_att_coordinates, \
                original_svg_points, svg_path_att_mth_candidate = generate_mutant(
                input_reshape_images_reverse(digit_reshaped_1), svg_path_att_mth, ext_att, SQUARE_SIZE, NUMBER_OF_POINTS,
                True, ATTENTION_METHOD)

            # If there is no highest attention point found, it means the digit mutated is close to an invalid digit
            # and we can stop mutating
            if list_of_svg_points is None:
                break

            list_of_svg_points_2 = list_of_svg_points
            shape = mutant_digit_att_candidate.shape

            # check if the candidate is good
            pred_class_mutant_att_candidate = model.predict(mutant_digit_att_candidate)
            pred_input_mutant_att_candidate = np.argmax(model.predict(mutant_digit_att_candidate), axis=-1)
            fitness_mutant_att_candidate = evaluate_ff2(pred_class_mutant_att_candidate, label)

            if fitness_mutant_att_candidate <= fitness_mutant_att:

                if ADAPTIVE:
                    if fitness_mutant_att_candidate < (0.99 * fitness_mutant_att):
                        print("RESETTING ext_att")
                        ext_att = EXTENT_LOWERBOUND
                        number_of_times_fitness_function_does_not_change_att = 0

                pred_input_mutant_att = pred_input_mutant_att_candidate
                pred_class_mutant_att = pred_class_mutant_att_candidate
                fitness_mutant_att = fitness_mutant_att_candidate
                mutant_digit_att = mutant_digit_att_candidate
                svg_path_att_mth = svg_path_att_mth_candidate
                digit_reshaped_1 = mutant_digit_att
            else:
                number_of_times_fitness_function_does_not_change_att += 1

                if ADAPTIVE:
                    if number_of_times_fitness_function_does_not_change_att > 10:
                        if (ext_att + EXTENT_STEP) <= EXTENT_UPPERBOUND:
                            ext_att = ext_att + EXTENT_STEP
                        number_of_times_fitness_function_does_not_change_att = 0

            mutated_points_att_list.append(point_mutated)
        else:
            mutated_points_att_list.append("NA")

        # if a misclassification is found we stop mutating
        if not miss_classification_found_normal:

            svg_path_normal_mth = get_svg_path(input_reshape_images_reverse(digit_reshaped_2)[0])
            mutant_digit_normal = digit_reshaped_2
            pred_class_mutant_normal = model.predict(digit_reshaped_2)
            fitness_mutant_normal = evaluate_ff2(pred_class_mutant_normal, label)
            pred_input_mutant_normal = np.argmax(model.predict(mutant_digit_normal), axis=-1)

            mutant_digit_normal_candidate, point_mutated_normal, svg_path_normal_mth_candidate = generate_mutant(
                input_reshape_images_reverse(digit_reshaped_2), svg_path_normal_mth, ext_normal, SQUARE_SIZE,
                NUMBER_OF_POINTS, False, ATTENTION_METHOD)

            pred_class_mutant_normal_candidate = model.predict(mutant_digit_normal_candidate)
            fitness_mutant_normal_candidate = evaluate_ff2(pred_class_mutant_normal_candidate, label)
            pred_input_mutant_normal_candidate = np.argmax(model.predict(mutant_digit_normal_candidate), axis=-1)

            if fitness_mutant_normal_candidate <= fitness_mutant_normal:
                pred_input_mutant_normal = pred_input_mutant_normal_candidate
                pred_class_mutant_normal = pred_class_mutant_normal_candidate
                fitness_mutant_normal = fitness_mutant_normal_candidate
                mutant_digit_normal = mutant_digit_normal_candidate
                svg_path_normal_mth = svg_path_normal_mth_candidate
                digit_reshaped_2 = mutant_digit_normal
            mutated_points_normal_list.append(point_mutated_normal)
        else:
            mutated_points_normal_list.append("NA")

        # Appending all the data to the list. Necessary to generate the plots of mutations sequentially.
        iteration_list.append(iteration)
        fitness_function_att.append(fitness_mutant_att)
        prediction_function_att.append(pred_class_mutant_att[0][label])
        fitness_function_normal.append(fitness_mutant_normal)
        prediction_function_normal.append(pred_class_mutant_normal[0][label])
        mutant_digit_att_list.append(mutant_digit_att)
        mutant_digit_normal_list.append(mutant_digit_normal)
        xai_images_list.append(xai)
        list_of_svg_points_list.append(list_of_svg_points)
        list_of_svg_points_list_2.append(list_of_svg_points_2)
        pred_input_mutant_att_list.append(pred_input_mutant_att[0])
        pred_input_mutant_normal_list.append(pred_input_mutant_normal[0])
        square_att_coordinates_list.append(square_att_coordinates)
        original_svg_points_list.append(original_svg_points)
        mutated_points_att_list_numeric.append(point_mutated)
        ext_att_list.append(ext_att)
        ext_normal_list.append(ext_normal)

        # Checking if the prediction of the mutant digit generated by ATTENTION Method is different from the ground truth (label)
        if pred_input_mutant_att[0] != label:
            if not miss_classification_found_att:
                iterations_detection_att_list.append(iteration)
                iterations_detection_att = iteration

                # Writing data to the stats.csv file - Data with the predicitions of both mutated digits (NORMAL and ATTENTION), Label and iteration
                with open(csv_path, "a") as f1:
                    writer = csv.writer(f1)
                    writer.writerow([image_index, "ATTENTION", METHOD, label, pred_input_mutant_att[0],
                                     pred_class_mutant_att[0][label], iteration])
            miss_classification_found_att = True

            if method_winner is None:
                method_winner = "Heatmaps"

        # Checking if the prediction of the mutant digit generated by NORMAL Method is different from the ground truth (label)
        if pred_input_mutant_normal[0] != label:
            if not miss_classification_found_normal:
                iterations_detection_normal_list.append(iteration)
                iterations_detection_normal = iteration

                # Writing data to the stats.csv file - Data with the predicitions of both mutated digits (NORMAL and ATTENTION), Label and iteration
                with open(csv_path, "a") as f1:
                    writer = csv.writer(f1)
                    writer.writerow([image_index, "NORMAL", METHOD, label, pred_input_mutant_normal[0],
                                     pred_class_mutant_normal[0][label], iteration])
            miss_classification_found_normal = True

            if method_winner is None:
                method_winner = "Normal"

        # If miss classifications were found for both mutation method, we can stop the loop
        if miss_classification_found_att and miss_classification_found_normal:
            method_winner = "Both"
            break
        else:
            method_winner = "None"

    # If True -> save the history of mutated digits independently whether it found a miss classification or not
    if True:
        if SAVE_IMAGES and list_of_svg_points is not None:
            folder_path = create_folder(DST, NUMBER_OF_MUTATIONS, repetition, ext_att, ext_normal, label,
                                        image_index, METHOD, "ATT_vs_NOR", run_id, seed,
                                        indices_chosen[image_index])
            # BROKEN
            save_images(mutant_digit_normal_list, mutant_digit_att_list, xai_images_list, list_of_svg_points_list,
                        iteration_list, fitness_function_att, prediction_function_att, fitness_function_normal,
                        prediction_function_normal, number_of_mutations, folder_path, pred_input_mutant_normal_list,
                        pred_input_mutant_att_list, ATTENTION_METHOD, square_size, square_att_coordinates_list,
                        original_svg_points_list, mutated_points_att_list_numeric, ext_att_list, ext_normal_list)
            make_gif(folder_path, folder_path + "/gif")

        # Writing data to the stats_2.csv file - Data reagarding a cycle of mutations.
        # iterations_detection_att -> The number of iterations ATTENTION method took to find a missclassification
        # iterations_detection_normal -> The number of iterations NORAML method took to find a missclassification
        # method_winner -> Which method took less iterations to find a missclassification
        with open(csv_path_2, "a") as f1:
            writer = csv.writer(f1)
            writer.writerow(
                [image_index, label, repetition, seed, iterations_detection_att, iterations_detection_normal,
                 method_winner])

        if SAVE_STATS4_CSV:
            # Writing the points mutated to the .csv
            list_to_write = []
            for iter in range(len(iteration_list)):
                list_to_write.append(
                    [iteration_list[iter], mutated_points_att_list[iter], mutated_points_normal_list[iter],
                     list_of_svg_points_list_2[iter][0]])

            with open(csv_path_4, "a") as f1:
                writer = csv.writer(f1)
                writer.writerows(list_to_write)
    else:
        with open(csv_path_2, "a") as f1:
            writer = csv.writer(f1)
            writer.writerow([image_index, label, REPETITION, seed, "NA", "NA", "Not Found"])

# Calculating averages and std dev
number_of_miss_classification_att = len(iterations_detection_att_list)
number_of_miss_classification_normal = len(iterations_detection_normal_list)
iterations_mean_att = "NA"
iterations_mean_normal = "NA"
iterations_std_att = "NA"
iterations_std_normal = "NA"

if number_of_miss_classification_normal != 0:
    iterations_mean_normal = np.mean(np.array(iterations_detection_normal_list))
    iterations_std_normal = np.std(np.array(iterations_detection_normal_list))
    iterations_mean_normal_list.append(iterations_mean_normal)
    number_of_miss_classification_normal_list.append(number_of_miss_classification_normal)
else:
    iterations_mean_normal_list = []
    number_of_miss_classification_normal_list = []

if number_of_miss_classification_att != 0:
    iterations_mean_att = np.mean(np.array(iterations_detection_att_list))
    iterations_std_att = np.std(np.array(iterations_detection_att_list))
    iterations_mean_att_list.append(iterations_mean_att)
    number_of_miss_classification_att_list.append(number_of_miss_classification_att)
else:
    iterations_mean_att_list = []
    number_of_miss_classification_att_list = []

# Writing data to the stats_3.csv file
with open(csv_path_3, "a") as f1:
    writer = csv.writer(f1)
    writer.writerow(
        [image_index, label, iterations_mean_att, iterations_mean_normal, iterations_std_att, iterations_std_normal,
         number_of_miss_classification_att, number_of_miss_classification_normal])

save_boxPlots(iterations_mean_att_list, iterations_mean_normal_list, number_of_miss_classification_att_list,
              number_of_miss_classification_normal_list, DST, EXTENT, NUMBER_OF_MUTATIONS, NUMBER_OF_REPETITIONS)
print("iterations_mean_att_list: ", iterations_mean_att_list)
print("iterations_mean_normal_list: ", iterations_mean_normal_list)
print("number_of_miss_classification_att_list: ", number_of_miss_classification_att_list)
print("number_of_miss_classification_normal_list: ", number_of_miss_classification_normal_list)
