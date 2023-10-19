import numpy as np


def orientation(p, q, r):
    val = np.dot(q - p, r - q)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def next_hull_point(points, p):
    q = p
    for r in points:
        o = orientation(p, q, r)
        if o == 2:
            q = r
    return q


def chan_algorithm(points, dim):
    n = len(points)
    if n <= 3:
        return points

    m = n // 2
    lower_hull = chan_algorithm(points[:m], dim)
    upper_hull = chan_algorithm(points[m:], dim)

    return lower_hull + upper_hull[1:]


def convex_hull_chan(points):
    dim = len(points)
    points = np.array(points)
    min_point = points[np.lexsort(points.T)]
    max_point = min_point[-1]
    min_point = min_point[0]

    lower = [min_point] + [point for point in points if
                           not np.array_equal(point, min_point) and not np.array_equal(point,
                                                                                       max_point) and orientation(
                               min_point, max_point, point) != 2]
    upper = [max_point] + [point for point in points if
                           not np.array_equal(point, min_point) and not np.array_equal(point,
                                                                                       max_point) and orientation(
                               min_point, max_point, point) != 1]

    upper_hull = chan_algorithm(upper, dim)
    lower_hull = chan_algorithm(lower, dim)

    return upper_hull + lower_hull[1:]
