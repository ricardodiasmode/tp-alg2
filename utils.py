import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score

import chan
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    return (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1]))


def intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def envelop_convex_overlap(first_hull, second_hull):
    n = len(first_hull)
    m = len(second_hull)

    for i in range(n):
        p1 = first_hull[i]
        q1 = first_hull[(i + 1) % n]

        for j in range(m):
            p2 = second_hull[j]
            q2 = second_hull[(j + 1) % m]

            if intersect(p1, q1, p2, q2):
                return True

    return False


def pre_process_data(all_data, all_class):
    processed_data = []
    processed_class = []

    first_found_class = all_class[0]
    i = 1
    while all_class[i] == first_found_class:
        i += 1
    second_found_class = all_class[i]

    for i in range(len(all_class)):
        if all_class[i] != first_found_class and all_class[i] != second_found_class:
            continue
        processed_class.append(all_class[i])
        processed_data.append(all_data[i])

    svd = TruncatedSVD(n_components=2)
    processed_data = [image.flatten() for image in processed_data]
    processed_data = np.array(processed_data)
    imputer = SimpleImputer(strategy='mean')
    processed_data = imputer.fit_transform(processed_data)
    processed_data = svd.fit_transform(processed_data)

    return processed_data, processed_class, first_found_class, second_found_class


def create_convex_hull(training_data, training_class, first_class_index):
    first_class_points = []
    second_class_points = []

    for i in range(len(training_class)):
        if training_class[i] == first_class_index:
            first_class_points.append(training_data[i])
        else:
            second_class_points.append(training_data[i])

    if first_class_points == [] or second_class_points == []:
        return None, None

    first_class_hull = chan.convex_hull_chan(first_class_points)
    second_class_hull = chan.convex_hull_chan(second_class_points)

    return first_class_hull, second_class_hull


def find_line_between_hulls(first_class_hull, second_class_hull, should_plot):
    points_pair = [(p1, p2) for p1 in first_class_hull for p2 in second_class_hull if
                    (p1[0] - p2[0] != 0) and (p1[1] - p2[1] != 0)]

    if len(points_pair) == 0:
        return None, None

    first_point, second_point = min(points_pair, key=lambda par: np.linalg.norm(np.array(par[0]) - np.array(par[1])))
    offset = ((first_point[0] + second_point[0]) / 2, (first_point[1] + second_point[1]) / 2)

    num = (second_point[1] - first_point[1])
    den = (second_point[0] - first_point[0])
    m = -(1 / (num / den))

    if should_plot:
        plot_everything(first_class_hull, second_class_hull, m, offset, first_point, second_point)

    return m, offset


def is_point_above(a, m, b):
    x_a, y_a = a
    x_b, y_b = b
    line_y = m * (x_a - x_b) + y_b
    return y_a > line_y


def plot_hull(points, hull_x, hull_y, color):
    if len(points) > 2:
        points = np.array(points)
        hull = ConvexHull(points)
        x = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
        y = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
        plt.plot(x, y, f'{color}-')
    else:
        plt.fill(hull_x, hull_y, color=f'{color}', alpha=0.2)


def plot_everything(first_hull, second_hull, m=None, middle_point=None, first_hull_closest_point=None, second_hull_closest_point=None):
    first_hull_x = [point[0] for point in first_hull]
    first_hull_y = [point[1] for point in first_hull]

    second_hull_x = [point[0] for point in second_hull]
    second_hull_y = [point[1] for point in second_hull]

    plot_hull(first_hull, first_hull_x, first_hull_y, 'b')
    plot_hull(second_hull, second_hull_x, second_hull_y, "r")

    plt.scatter(first_hull_x, first_hull_y, color='blue')
    plt.scatter(second_hull_x, second_hull_y, color='red')

    if m is not None and middle_point is not None:
        min_x_lin = min(first_hull_x + second_hull_x)
        max_x_lin = max(first_hull_x + second_hull_x)
        line_x = np.linspace(min_x_lin, max_x_lin, 100)
        line_y = (m * (line_x - middle_point[0])) + middle_point[1]
        plt.plot(line_x, line_y, color='green', label=f'y = {m:.{2}f}(x - {middle_point[0]:.{2}f}) + {middle_point[1]:.{2}f}')

        x1, y1 = first_hull_closest_point
        x2, y2 = second_hull_closest_point
        x = [x1, x2]
        y = [y1, y2]
        plt.plot(x, y, marker='o', color='black')
        plt.legend()

    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.show()


def use_data(data_unfiltered, class_unfiltered, should_plot):
    print("=== Starting new classification ===")
    # Variable setups
    sorted_indexes = np.arange(len(data_unfiltered))
    np.random.shuffle(sorted_indexes)
    data_sorted = data_unfiltered[sorted_indexes]
    class_sorted = class_unfiltered[sorted_indexes]

    # Pre-processing data and getting training variables
    all_data, all_class, first_used_class, second_used_class = pre_process_data(data_sorted, class_sorted)
    training_data = all_data[:int(0.7 * len(all_data))]
    training_class = all_class[:int(0.7 * len(all_class))]

    # Getting hulls
    first_class_hull, second_class_hull = create_convex_hull(training_data, training_class, first_used_class)
    if second_class_hull is None or first_class_hull is None:
        print("Could not create a convex hull.")
        print("=== Classification finished ===")
        return

    if envelop_convex_overlap(first_class_hull, second_class_hull):
        if should_plot:
            plot_everything(first_class_hull, second_class_hull)
        print("The convex hull has some intersection.")
        print("=== Classification finished ===")
        return

    # Getting line
    m, offset = find_line_between_hulls(first_class_hull, second_class_hull, should_plot)
    if m is None:
        if should_plot:
            plot_everything(first_class_hull, second_class_hull)
        print("Impossible to find a line between convex hulls.")
        print("=== Classification finished ===")
        return  # If failed to get line makes no sense to try to classify

    # Checking if the first hull is above the line
    test_data = all_data[-int(0.3 * len(all_class)):]
    test_class = all_class[-int(0.3 * len(all_class)):]
    first_is_above = is_point_above(first_class_hull[0], m, offset)

    # Classifying
    test_result = []
    for i in range(len(test_data)):
        current_test_data = test_data[i]
        point_loc = is_point_above(current_test_data, m, offset)
        if point_loc == first_is_above:
            test_result.append(first_used_class)
        else:
            test_result.append(second_used_class)

    # Checking the accuracy and the other metrics
    points = 0
    for i in range(len(test_class)):
        if test_class[i] == test_result[i]:
            points += 1
    accuracy = points / len(test_class)
    recall = recall_score(test_class, test_result, pos_label=test_class[0])
    f1 = f1_score(test_class, test_result, pos_label=test_class[0])
    print(f'Classifier accuracy: {accuracy}')
    print(f'Classifier recall: {recall}')
    print(f'Classifier f1-score: {f1}')
    print("=== Classification finished ===")
