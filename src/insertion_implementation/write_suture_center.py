import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


def reverse(line):
    """

    :param line: {numpy.array} -- the 2d line need to be flipped only on column 1
    :return: line: {numpy.array} -- already flipped line
    """
    left = 0
    right = line.shape[0] - 1
    while left < right:
        line[left][0], line[right][0] = line[right][0], line[left][0]
        line[left][1], line[right][1] = line[right][1], line[left][1]
        left += 1
        right -= 1
    return line


def rotate_line(line):
    """

    :param line: {numpy.array} -- the array of points list
    :return: rotated_line: {numpy.array} -- the array of points list
    :return: angle: {float} -- the rotation angle
    :return: origin: {numpy.array} -- the origin of the rotation
    """
    origin = np.array([(line[0][0]+line[-1][0])/2, (line[0][1]+line[-1][1])/2])
    tan = (line[0][1] - origin[1])/(origin[0]-line[0][0])
    angle = np.arctan(tan)
    rotated_line = np.zeros(line.shape)
    for i in range(line.shape[0]):
        rotated_line[i][0] = origin[0] + np.cos(angle) * (line[i][0] - origin[0]) - np.sin(angle) * (
                    line[i][1] - origin[1])
        rotated_line[i][1] = origin[1] + np.sin(angle) * (line[i][0] - origin[0]) + np.cos(angle) * (
                    line[i][1] - origin[1])
    return rotated_line, angle, origin


def cal_gradient(line, dis):
    """

    :param line: {numpy.array} -- the array of points in the line
    :param dis: {int} -- the distance to calculate the gradient
    :return:
    """
    start = int(line.shape[0] * (0.5 / 10))
    end = int(line.shape[0] * (9.5 / 10))
    mod_line = line[start:end, :]
    pre_gra_line = line[start:, :]
    grad_list = []
    for i in range(mod_line.shape[0] - dis):
        grad = np.zeros(2)
        grad[0] = mod_line[i + dis][0]
        if (pre_gra_line[i + dis][0] - pre_gra_line[i][0]) != 0:
            grad[1] = float(pre_gra_line[i + dis][1] - pre_gra_line[i][1]) / (
                        pre_gra_line[i + dis][0] - pre_gra_line[i][0])
        else:
            grad[1] = grad_list[-1][1]
        grad_list.append(grad)
    grad_list = np.array(grad_list)
    return mod_line, grad_list


def lowest_ind(line):
    """

    :param line: {numpy.array} -- the 2d line need to be found lowest point
    :return: index: {int} -- the index of the lowest point
    """
    index = 0
    lowest_val = 5000
    for i in range(line.shape[0]):
        if line[i][1] <= lowest_val:
            lowest_val = line[i][1]
            index = i
    return index


def rotate_back(line, angle, origin):
    """

    :param line: {numpy.array} -- the array of points list
    :param angle: {float} -- the rotation angle
    :param origin: {numpy.array} -- the origin of the rotation
    :return: rotated_line: {numpy.array} -- the array of points list
    """
    angle = -angle
    rotated_line = np.zeros(line.shape)
    for i in range(line.shape[0]):
        rotated_line[i][0] = origin[0] + np.cos(angle) * (line[i][0] - origin[0]) - np.sin(angle) * (
                    line[i][1] - origin[1])
        rotated_line[i][1] = origin[1] + np.sin(angle) * (line[i][0] - origin[0]) + np.cos(angle) * (
                    line[i][1] - origin[1])
    return rotated_line


def find_wound(grad_lis, line, dis):
    """

    :param grad_lis: {numpy.array} -- the array of gradient list
    :param line: {numpy.array} -- the 2d line need to be found wound position
    :param dis:{int} -- the distance to calculate gradient
    :return: left_ind: {int} -- wound start point index
    :return: right_ind: {int} -- wound right point index
    """
    points = np.zeros((2, 2))
    find_st = False
    find_en = False
    left_ind = None
    right_ind = None
    for i in range(grad_lis.shape[0]):
        if grad_lis[i][1] < -1.5:
            find_st = True
            points[0] = line[i+dis]
            left_ind = i+dis
            break
    for i in range(grad_lis.shape[0]):
        if grad_lis[grad_lis.shape[0]-i-1][1] > 1.5:
            find_en = True
            points[1] = line[grad_lis.shape[0]-i-1+dis]
            right_ind = grad_lis.shape[0]-i-1+dis
            break
    if find_st is True and find_en is True and points[0][0] < points[1][0]:
        return left_ind, right_ind
    else:
        return None, None


def cal_distance(mod_line, start_p, end_p, pix_x, pix_h):
    """

    :param mod_line: {numpy.array} -- the array of points that draw the top of the wound
    :param start_p: {int} -- the index of the start point
    :param end_p: {int} -- the index of the end of point
    :param pix_x: {float} -- the length in pixels x length
    :param pix_h: {float} -- the length in pixels y length
    :return: dis_list: {numpy.array} -- the distance list between pixel and pixel
    :return: distance: {float} -- the distance between two point
    """
    dis_list = []
    distance = 0
    for i in range(start_p, end_p):
        sub_dis = np.sqrt(((mod_line[i][0]-mod_line[i+1][0])*pix_x)**2+((mod_line[i][1]-mod_line[i+1][1])*pix_h)**2)
        distance += sub_dis
        dis_list.append(sub_dis)
    return np.array(dis_list), distance


def find_origin(down_p_pix, right_p_pix, rad, pix_x, pix_h):
    """

    :param down_p_pix: {np.array} -- the deepest point of the suture path
    :param right_p_pix: {np.array} -- the bite point of the suture path at the first part of suture
    :param rad: {float} -- the number of pixels of the radius
    :param pix_x: {float} -- the length in pixels x length
    :param pix_h: {float} -- the length in pixels y length
    :return: origin: {np.array} -- the suture origin
    """
    down_p = np.array([down_p_pix[0]*pix_x, down_p_pix[1]*pix_h])
    right_p = np.array([right_p_pix[0]*pix_x, right_p_pix[1]*pix_h])
    # axis distance between two point
    center = np.array([right_p[0]-down_p[0], right_p[1]-down_p[1]])
    # distance between two points
    point_dis = np.sqrt(center[0]**2 + center[1]**2)
    half_p = np.array([(right_p[0]+down_p[0])/2, (right_p[1]+down_p[1])/2])
    # distance alone the mirror line
    mir_dis = np.sqrt(rad**2-(point_dis/2)**2)
    origin_dis = np.array([half_p[0]-mir_dis*center[1]/point_dis,
                      half_p[1]+mir_dis*center[0]/point_dis])
    origin = np.array([int(origin_dis[0]/pix_x), int(origin_dis[1]/pix_h)])
    return origin


def write_suture_center(serial_num, ideal_cscan):
    radius = 4.205
    dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                   serial_num + "/dimension.csv", header=None))
    pix_x = dim_arr[0][0] / dim_arr[1][0]
    pix_h = dim_arr[0][1] / dim_arr[1][1]
    raw_line = np.array(pd.read_csv("../../data/suture_experiment/segmentation_line_files/" +
                                    serial_num + "_line.csv", header=None))
    raw_line = np.multiply(np.array([1, -1]), raw_line)
    raw_line = reverse(raw_line)
    raw_line, angle, origin = rotate_line(raw_line)
    gau_line_raw = np.zeros(raw_line.shape)
    gau_line_raw[:, 0] = raw_line[:, 0]
    gau_line_raw[:, 1] = gaussian_filter1d(raw_line[:, 1], 8)
    mod_line_r, grad_list = cal_gradient(raw_line, 10)
    gau_line_r, grad_list_g = cal_gradient(gau_line_raw, 10)
    low_p = lowest_ind(mod_line_r)
    left_ind, right_ind = find_wound(grad_list_g, mod_line_r, 10)
    mod_line = rotate_back(mod_line_r, angle, origin)
    lef_dis_list, lef_dis = cal_distance(mod_line, left_ind, low_p, pix_x, pix_h)
    rig_dis_list, rig_dis = cal_distance(mod_line, low_p, right_ind, pix_x, pix_h)
    searching_length = 0.20 * min(lef_dis, rig_dis)
    left_sub_dis = 0
    left_bite_pix = 0
    for i in range(low_p - left_ind):
        left_sub_dis += lef_dis_list[-i - 1]
        if left_sub_dis >= searching_length:
            left_bite_pix = i
            break
    right_sub_dis = 0
    right_bite_pix = 0
    for i in range(right_ind - low_p):
        right_sub_dis += rig_dis_list[i]
        if right_sub_dis >= searching_length:
            right_bite_pix = i
            break
    left_bitepoint = mod_line[low_p - left_bite_pix]
    right_bitepoint = mod_line[low_p + right_bite_pix]
    suture_origin = find_origin(left_bitepoint, right_bitepoint, radius, pix_x, pix_h)
    np.savetxt("../../data/suture_center_files/" + serial_num + "_suture_center.csv",
               np.array([suture_origin[0], suture_origin[1], ideal_cscan]))


if __name__ == "__main__":
    serial_num = '190911A'
    ideal_scsan = 400
    write_suture_center(serial_num, ideal_scsan)
