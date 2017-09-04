"""
Collection of once-used functions I may be using one day again
"""

def rgb_to_hex(red, green, blue):
    """
    Convert (int, int, int) RGB to hex RGB
    """
    return '#%02x%02x%02x' % (red, green, blue)


def dist(a, b):
    """
    Euclidian distance
    """
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**.5


def find_nearest(xy, all_data):
    """
    Finds the nearest point in all_data (not optimized!)

    :type all_data: array( (float,float) )
    :return: (x_nearest,y_nearest), index_nearest
    :rtype: (float, float), int
    """
    x, y = xy
    best_distance = float('inf')
    best_point = None
    idx = 0

    for i, point in enumerate(all_data):
        new_dist = dist([x, y], point)
        if best_distance > new_dist:
            best_point = point
            best_distance = new_dist
            idx = i

    return point, idx

