import random
import re
from os import makedirs
from os.path import exists
from random import choice

import matplotlib
from tensorflow import keras

from attention_maps import gradcam

matplotlib.use('Agg')

from tf_keras_vis.utils.scores import CategoricalScore
from keras.applications.imagenet_utils import preprocess_input

import time

from operator import itemgetter

import vectorization_tools
from rasterization_tools import *
from vectorization_tools import *

matplotlib.use('Agg')

import numpy as np

import rasterization_tools
from config import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB

import potrace

from random import randint, uniform


def get_svg_path(image):
    array = preprocess(image)
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    return createSVGpath(path)


def input_reshape_images_reverse(x):
    x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape *= 255.0
    return x_reshape


def generate_mutant(image, svg_path, extent, square_size, number_of_points, mutation_method, ATTENTION_METHOD):
    if mutation_method:
        if ATTENTION_METHOD == "mth5":
            mutante_digit_path, list_of_svg_points, xai, point_mutated = apply_mutoperator_attention(image, svg_path,
                                                                                                     extent, square_size,
                                                                                                     number_of_points)
        elif ATTENTION_METHOD == "mth1":
            mutante_digit_path, list_of_svg_points, xai, point_mutated = apply_mutoperator_attention_2_1(image, svg_path,
                                                                                                         extent,
                                                                                                         square_size)
        elif ATTENTION_METHOD == "distances":
            mutante_digit_path, list_of_svg_points, xai, point_mutated, square_att_coordinates, original_svg_points = apply_mutoperator_attention_distance_mth(
                image, svg_path, extent, square_size, number_of_points)
        elif ATTENTION_METHOD == "probability":
            mutante_digit_path, list_of_svg_points, xai, point_mutated, square_att_coordinates, original_svg_points = apply_mutoperator_attention_roullet(
                image, svg_path, extent, square_size, number_of_points)
        # print(mutante_digit_path)
        if list_of_svg_points != None and ("C" in mutante_digit_path) and ("M" in mutante_digit_path):
            rast_nparray = rasterize_in_memory(create_svg_xml(mutante_digit_path))
            return rast_nparray, list_of_svg_points, xai, point_mutated, square_att_coordinates, original_svg_points, mutante_digit_path
        else:
            return image, None, xai, None, None, original_svg_points, None
    else:
        mutante_digit_path, point_mutated = apply_mutoperator2(image, svg_path, extent)
        rast_nparray = rasterization_tools.rasterize_in_memory(vectorization_tools.create_svg_xml(mutante_digit_path))
        # print("original_mutated_digit shape", rast_nparray.shape)
        # print("original_mutated_digit max", rast_nparray.max())
        # print("original_mutated_digit min", rast_nparray.min())
        return rast_nparray, point_mutated, mutante_digit_path


def evaluate_ff2(predictions, lbl):
    predictions = predictions.tolist()
    prediction = predictions[0]
    confidence_expclass = prediction[lbl]
    # print("confidence_expclass", confidence_expclass)
    unexpected_prediction = prediction[0:lbl] + prediction[lbl + 1:10]
    # print("unexpected_prediction", unexpected_prediction)
    confidence_unexpectclass = max(unexpected_prediction)
    fitness = confidence_expclass - confidence_unexpectclass

    return fitness


def input_reshape_and_normalize_images(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape


def create_folder(mutant_root_folder, number_of_mutations, repetition, ext_att, ext_normal, label, image_index, method,
                  attention, run_id, seed, mnist_index):
    # run_id_2 = str(Timer.start.strftime('%s'))
    # DST = "mutants/debug/debug_"+ Mth1_str + run_id +"/NM="+ str(number_of_mutations) + "_REP=" + str(repetition) + "_ext="+str(extent)+"_lbl="+str(label)+"_IMG_INDEX="+str(image_index)+"_mth="+method+"_ATT="+str(attention)#+"_run_"+str(run_id_2)
    # DST = mutant_root_folder +"/NM="+ str(number_of_mutations) + "_REP=" + str(repetition) + "_ext="+str(extent)+"_lbl="+str(label)+"_IMG_INDEX="+str(image_index)+"_mth="+method+"_ATT="+str(attention)#+"_run_"+str(run_id_2)
    DST = mutant_root_folder + "/IMG=" + str(image_index) + "_INDEX=" + str(mnist_index) + "_Seed=" + str(
        seed) + "_REP=" + str(repetition) + "_lbl=" + str(label)
    # DST = mutant_root_folder + "/IMG_INDEX="+str(image_index) + "_Seed=" + str(seed) + "_REP=" + str(repetition) + "_lbl="+str(label) + "_ext_att=" + str(ext_att) + "_ext_normal=" + str(ext_normal)
    if not exists(DST):
        makedirs(DST)

    return DST
    # DST_ARC = join(DST, "archive")
    # DST_IND = join(DST, "inds")


def initializate_list_of_images(images, labels, number_of_samples, specific_indexes=None):
    if specific_indexes is None:
        list_of_indices = []
        for i in range(number_of_samples):
            for label in reversed(range(0, 10)):
                indices = np.where(labels == label)
                indice = choice(indices[0])
                list_of_indices.append(indice)

        print("list_of_indices randomly chosen: ", list_of_indices)
        print("Respective Labels: ", labels[list_of_indices])

        return images[list_of_indices], labels[list_of_indices], list_of_indices
    else:
        if isinstance(specific_indexes, str):
            specific_indexes = int(specific_indexes)
        if isinstance(specific_indexes, int):
            list_of_indices = [specific_indexes]
        else:
            list_of_indices = specific_indexes
        print("list_of_indices randomly chosen: ", list_of_indices)
        print("Respective Labels: ", labels[list_of_indices])
        return images[list_of_indices], labels[list_of_indices], list_of_indices


def apply_mutoperator2(input_img, svg_path, extent):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)

    # chose a random control point
    num_matches = len(segments) * 4
    path = svg_path
    if num_matches > 0:
        random_coordinate_index = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        control_point = next(
            value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 4)))
        # print("control_point", control_point)
        group_index = (random_coordinate_index % 4) + 1
        # print("control_point.group(group_index)", control_point.group(group_index))
        # print("group_index", group_index)
        value = apply_displacement_to_mutant(control_point.group(group_index), extent)
        path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
        return path, control_point
    else:
        print("ERROR")
        print(svg_path)
        return path, "Error"


def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


def apply_mutoperator_attention_distance_mth(input_img, svg_path, extent, square_size, number_of_points):
    list_of_points_close_to_square_attention_patch, elapsed_time, xai, square_att_coordinates, original_svg_points = get_svg_points_distance_mth(
        input_img, square_size, square_size,
        svg_path, number_of_points)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    # if len(list_of_points_close_to_square_attention_patch) != 0:
    if list_of_points_close_to_square_attention_patch != None:
        original_point = random.choice(list_of_points_close_to_square_attention_patch)
        original_coordinate = random.choice(original_point)

        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)

        path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))

        # TODO: it seems that the points inside the square attention patch do not precisely match the point coordinates in the svg, to be tested
        return path, list_of_points_close_to_square_attention_patch, xai, original_point, square_att_coordinates, original_svg_points
    else:
        return svg_path, None, xai, None, None, None


def get_svg_points_distance_mth(images, x_patch_size, y_patch_size, svg_path, number_of_points):
    """
    get_svg_points_distance_mth Iterate all the image looking for the region with more attention and return list of SVG points (tuples) closest from those regions.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """
    # start_time1 = time.time()
    xai = compute_attention_maps(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsCloseToRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]
        if len(ControlPoints) != 0:
            x, y = get_attetion_region(xai[i], images[i], x_patch_size,
                                       y_patch_size)  # Getting coordinates of the highest attetion region (patch) reference point
            list_of_ControlPointsCloseToRegion = getControlPointsCloseToRegion(x, y, x_patch_size, y_patch_size,
                                                                               controlPoints,
                                                                               number_of_points)  # Getting all the points inside the highest attetion patch
            # list_of_ControlPointsCloseToRegion.append(ControlPointsCloseToRegion)
        else:
            return None, "(end_time - start_time1)", xai, None, controlPoints

    # end_time = time.time()

    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)
    # print("Find attention points time mth1: ", find_time)
    # print("Total time mth1: ", total_time)
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n")
    return list_of_ControlPointsCloseToRegion, "(end_time - start_time1)", xai, (x, y), controlPoints


def compute_attention_maps(images):  # images should have the shape: (x, 28, 28) where x>=1

    # start_time = time.time()
    images_reshaped = input_reshape_and_normalize_images(images)

    X = preprocess_input(images_reshaped, mode="tf")

    cam = gradcam(CategoricalScore(0),
                  X,
                  penultimate_layer=-1)

    # if Attention_Technique == "Faster-ScoreCAM":
    #
    #     # Generate heatmap with Faster-ScoreCAM
    #     cam = scorecam(score,
    #                    X,
    #                    penultimate_layer=-1,
    #                    max_N=10)
    #
    # elif Attention_Technique == "Gradcam++":
    #
    #     # Generate heatmap with GradCAM++
    #     cam = gradcam(score,
    #                   X,
    #                   penultimate_layer=-1)
    # else:
    #     print("Choose a valid attention technique")
    #     cam = None
    #     exit()

    return cam


def apply_mutoperator_attention(input_img, svg_path, extent, square_size, number_of_points):
    list_of_svg_points, elapsed_time, xai = AM_get_attetion_svg_points_images_v1(input_img, number_of_points,
                                                                                 svg_path, square_size)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    # print("list_of_svg_points", list_of_svg_points)
    if len(list_of_svg_points[0]) != 0:
        original_point = random.choice(list_of_svg_points[0])
        original_coordinate = random.choice(original_point)

        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)

        path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))

        # TODO: it seems that the points inside the square attention patch do not precisely match the point coordinates in the svg, to be tested
        return path, list_of_svg_points, xai, original_point
    else:
        return svg_path, None, xai, None


def AM_get_attetion_svg_points_images_v1(images, number_of_points, svg_path, sqr_size):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param number_of_points: Number of points (n) to return
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of n points (number_of_points) with more score attention around it. List of tuples Ex: (x,y)
    """
    # start_time1= time.time()
    xai = compute_attention_maps(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # position, elapsed_time = get_attetion_region_mth4(xai[i], controlPoints, sqr_size) #Getting coordinates of the highest attetion region (patch) reference point
        positions = get_SVG_points_with_sqr_attention(xai[i], controlPoints, sqr_size,
                                                      number_of_points)  # Getting coordinates of the highest attetion region (patch) reference point
        # print("positions", positions)

        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()
    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)
    # print("Find attention points time mth5: ", find_time)
    # print("Total time mth5: ", total_time)
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n")
    return positions, "(end_time - start_time)", xai


def get_SVG_points_with_sqr_attention(xai_image, svg_path_list, sqr_size, number_of_points):
    start_time = time.time()
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0

    max_sum_xai = 0
    # pos_max = svg_path_list[0]
    list_pos_and_values = []
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        list_pos_and_values.append([pos, sum_xai])
        if sum_xai > max_sum_xai:
            max_sum_xai = sum_xai
            pos_max = pos

    get_1 = itemgetter(1)
    list_pos_and_values_sorted = sorted(list_pos_and_values, key=get_1, reverse=True)
    list_to_return = list_pos_and_values_sorted[0:number_of_points]
    new_list = [item[0] for item in list_to_return]
    # print("MAXIMUM SUM_XAI =", sum_xai)
    end_time = time.time()
    # print("SUM XAI TEST:",sum_test)

    # Render XAI Images
    # f, ax = plt.subplots()
    # heatmap = np.uint8(cm.jet(xai_image)[..., :3] * 255)
    # ax.set_title("Time: " + str((end_time - start_time)))
    # ax.imshow(heatmap, cmap='jet')
    # ax.scatter(*zip(*svg_path_list),s=80)

    # for z, sum_value in enumerate(svg_path_list):
    #     ax.annotate("("+str(svg_path_list[z][0])+","+str(svg_path_list[z][1])+")", (svg_path_list[z][0], svg_path_list[z][1]))
    # plt.tight_layout()
    # plt.savefig("./xai/"+str(time.time())+"mth3_sqr=3_opt1.png")

    # plt.cla()

    return [new_list]  # , (end_time - start_time)


def getControlPointsCloseToRegion(x, y, x_patch_size, y_patch_size, controlPoints, number_of_points):
    listOfPointsAndDistances = []
    square_coordinate_X = x + x_patch_size / 2
    square_coordinate_Y = y + y_patch_size / 2
    square_coordinate = (square_coordinate_X, square_coordinate_Y)
    for point in controlPoints:
        # Calculate the distance between the SVG Point (point) and the coordinates of the area with highest attention
        dist = get_Euclidean_Distance(point, square_coordinate)
        listOfPointsAndDistances.append([point, dist])

    get_1 = itemgetter(1)
    list_pos_and_values_sorted = sorted(listOfPointsAndDistances, key=get_1, reverse=False)
    list_to_return = list_pos_and_values_sorted[0:number_of_points]
    new_list = [item[0] for item in list_to_return]
    # print(new_list)
    return new_list


def get_Euclidean_Distance(point1, point2):  # Point 1 and point 2 should be in this format (x,y) or [x,y]
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    result = ((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)

    return result


def get_attetion_region(xai_image, orig_image, x_sqr_size, y_sqr_size):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0
    x_final_pos = 0
    y_final_pos = 0

    for y_sqr_pos in range(y_dim - y_sqr_size):
        for x_sqr_pos in range(x_dim - x_sqr_size):
            sum_xai = 0
            for y_in_sqr in range(y_sqr_size):
                y_pixel_pos = y_sqr_pos + y_in_sqr
                for x_in_sqr in range(x_sqr_size):
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    # if orig_image[y_pixel_pos][x_pixel_pos] > 0:
                    sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
            if sum_xai > greater_value_sum_xai:
                greater_value_sum_xai = sum_xai
                x_final_pos = x_sqr_pos
                y_final_pos = y_sqr_pos

    return x_final_pos, y_final_pos


def apply_mutoperator_attention_2_1(input_img, svg_path, extent, square_size):
    list_of_points_inside_square_attention_patch, elapsed_time, xai = AM_get_attetion_svg_points_images_mth1_1(input_img,
                                                                                                               square_size,
                                                                                                               square_size,
                                                                                                               svg_path)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    if list_of_points_inside_square_attention_patch != None:

        list_of_mutated_coordinates_string = apply_displacement_to_mutant_2(
            list_of_points_inside_square_attention_patch[0], extent)

        path = svg_path
        list_of_points = list_of_points_inside_square_attention_patch[0]
        for original_coordinate_tuple, mutated_coordinate_tuple in zip(list_of_points,
                                                                       list_of_mutated_coordinates_string):
            original_coordinate = str(original_coordinate_tuple[0]) + "," + str(original_coordinate_tuple[1])
            # print("original coordinate", original_coordinate)
            # print("mutated coordinate", mutated_coordinate_tuple)
            path = path.replace(original_coordinate, mutated_coordinate_tuple)

        return path, list_of_points_inside_square_attention_patch, xai, list_of_points_inside_square_attention_patch
    else:
        return None, None, xai


def apply_mutoperator_attention_roullet(input_img, svg_path, extent, square_size, number_of_points):
    list_of_weights, list_of_probs, elapsed_time, xai, original_svg_points = AM_get_attetion_svg_points_images_prob(
        input_img, square_size, svg_path, number_of_points)

    if list_of_weights != None:
        original_point = random.choices(population=original_svg_points, weights=list_of_weights, k=1)[0]
        original_coordinate = random.choice(original_point)

        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)

        path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))

        # TODO: the points inside the square may not precisely match the point coordinates, to be tested
        return path, list_of_probs, xai, original_point, "NA", original_svg_points
    else:
        return svg_path, None, xai, None, None, None


def AM_get_attetion_svg_points_images_prob(images, square_size, svg_path, number_of_points):
    """
    AM_get_attetion_svg_points_images_mth2 Calculate the attetion score around each SVG path point and return a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param sqr_size: X and Y size of the square region
    :param model: The model object that will predict the value of the digit in the image
    :return: A a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """

    """
    #-----------------Structure of the list returned----------------#
     Start of the list -> [
                             image_0 (list)-> [
                                 point_0 of image_0 (list) -> [
                                     Position (x, y) of point_0 (tuple) -> (x0,y0),
                                     Weights for non-uniform distribution for point_0 (float) -> float
                                 ],
                                 point_1 of image_0 (list) -> [
                                     Position (x, y) of point_1 (tuple) -> (x1,y1),
                                     Weights for non-uniform distribution for point_1 (float) -> float
                                 ],
                                 [
                                     (x2,y2), float
                                 ],
                                 .
                                 .
                                 .
                                 [(xn,yn),float]
                             ],
                             image_1 (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]],  
                             image_2 (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]],
                               .
                               .
                               .
                             image_n (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]]        

     End of the list -> ]
    #----------------- END Structure of the list returned----------------#
    """
    xai = compute_attention_maps(images)
    start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    # list_of_weights = []
    # list_of_probs = []

    pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
    ControlPoints = pattern.findall(svg_path)
    controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]
    if len(ControlPoints) != 0:
        weight_list, prob_list = get_attetion_region_prob(xai[0], controlPoints, square_size)
    else:
        return None, "NA", xai, controlPoints
    # ControlPoints = vectorization_tools.getImageControlPoints(images[i])
    # print("image",i )

    end_time = time.time()
    # print("Find attention points time mth3: ", (end_time - start_time))
    return weight_list, prob_list, (end_time - start_time), xai, controlPoints


def get_attetion_region_prob(xai_image, svg_path_list, sqr_size):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0

    xai_list = []
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        xai_list.append(sum_xai)

    sum_xai_list = sum(xai_list)

    list_of_weights = []
    list_of_probabilities = []

    for sum_value, pos in zip(xai_list, svg_path_list):
        list_of_weights.append(np.exp((sum_value / sum_xai_list) * 100))

    sum_weights_list = sum(list_of_weights)
    for weight in list_of_weights:
        list_of_probabilities.append(weight / sum_weights_list)

    return list_of_weights, list_of_probabilities


def AM_get_attetion_svg_points_images_mth1_1(images, x_patch_size, y_patch_size, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """
    # start_time1 = time.time()
    xai = compute_attention_maps(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]
        if len(ControlPoints) != 0:
            x, y = get_attetion_region(xai[i], images[i], x_patch_size,
                                       y_patch_size)  # Getting coordinates of the highest attetion region (patch) reference point
            ControlPointsInsideRegion = getControlPointsInsideAttRegion(x, y, x_patch_size, y_patch_size,
                                                                        controlPoints)  # Getting all the points inside the highest attetion patch
            list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)
        else:
            return None, "(end_time - start_time1)", xai

    return list_of_ControlPointsInsideRegion, "(end_time - start_time1)", xai


def getControlPointsInsideAttRegion(x, y, x_dim, y_dim, controlPoints):
    list_of_points = []
    foundPoint = False
    for cp in controlPoints:
        if cp[0] >= (x - 1) and cp[0] < x + x_dim + 1:
            if cp[1] >= (y - 1) and cp[1] < y + y_dim + 1:
                list_of_points.append(cp)
                foundPoint = True
    if foundPoint == False:
        list_of_points.append(controlPoints[0])

    return list_of_points


def apply_displacement_to_mutant_2(list_of_points, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    x_or_y = random.choice((0, 1))
    y_or_x = (x_or_y - 1) * -1
    list_of_mutated_coordinates_string = []
    coordinate_matutated = [0, 0]
    for point in list_of_points:
        coordinate_matutated[y_or_x] = point[y_or_x]
        value = point[x_or_y]
        if random.uniform(0, 1) >= MUTOFPROB:
            result = float(value) + displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0]) + "," + str(coordinate_matutated[1]))
        else:
            result = float(value) - displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0]) + "," + str(coordinate_matutated[1]))

    return list_of_mutated_coordinates_string


def save_boxPlots(a, b, c, d, folder_path, ext, number_of_mutations, number_of_repetitions):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], height_ratios=[1])
    ax0 = fig.add_subplot(gs[0, 0])
    data0 = [a, b]
    bp0 = ax0.boxplot(data0, labels=["Attention", "Random"], patch_artist=True)
    ax0.grid(True)
    ax0.set_title("Number of iterations to mis-classification")

    colors = ['blue', 'red']

    for patch, color in zip(bp0['boxes'], colors):
        patch.set_facecolor(color)

    ax1 = fig.add_subplot(gs[0, 1])
    data1 = [c, d]
    bp1 = ax1.boxplot(data1, labels=["Attention", "Random"], patch_artist=True)
    ax1.grid(True)
    ax1.set_title("Number of mis-classifications")

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    # fig.suptitle("Ext="+str(ext)+"_#Mutations="+str(number_of_mutations)+"_#Repetitions="+str(number_of_repetitions))
    plt.tight_layout()
    plt.savefig(folder_path + "/boxplots" + "_Ext=" + str(ext) + "_#Mutations=" + str(
        number_of_mutations) + "_#Repetitions=" + str(number_of_repetitions) + ".png")
    plt.cla()
    plt.close(fig)
