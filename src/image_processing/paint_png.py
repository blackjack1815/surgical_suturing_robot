import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import os


# This function only generate suture path on the direction of Bscan
def draw_part_circle_3d(radius, center, start_Z, rot_angle, pix_x, pix_h, pix_z):
    """

    :param radius: {float} -- the radius of suture path
    :param center: {list} -- the suture center in the OCT frame
    :param start_Z: {float} -- the suture path Z axis value in the OCT frame
    :param rot_angle: {float} -- the rotation angle of the suture path
    :param start_angle: {float} -- the start suture angle
    :param rot_x: {float} -- the suture path rotation on the OCT frame X axis
    :param pix_x: {float} -- the length in pixels x length
    :param pix_h: {float} -- the length in pixels y length
    :param pix_z: {float} -- the length in pixels z length
    :return: suture_path: {list} -- the suture milestones in the world frame
    """
    suture_path = []
    for i in range(0, rot_angle*10):
        current_angle = i*0.1
        pos_x_0 = center[0] - radius * np.cos((current_angle * np.pi) / 180.)
        pos_y_0 = center[1] + radius * np.sin((current_angle * np.pi) / 180.)
        milestone = np.divide([pos_x_0, pos_y_0, start_Z], [pix_x, pix_h, pix_z])
        suture_path.append(milestone)
    return suture_path


def paint_bscan(bscan, serial_num, idx=None, circle_1=None, circle_2=None, circle_3=None, line=None):
    bscan_color = cv.cvtColor(bscan.astype(np.uint8), cv.COLOR_GRAY2BGR)
    bscan_color = cv.flip(bscan_color, 1)
    color_1 = [0, 255, 0]
    color_2 = [0, 0, 255]
    color_3 = [255, 0, 0]
    color_4 = [0, 255, 255]
    if line is not None:
        for i in range(line.shape[0]):
            bscan_color[int(line[i][1]), int(line[i][0])] = color_1
            # bscan_color[int(line[i][1]+1), int(line[i][0])] = color_1
            # bscan_color[int(line[i][1]-1), int(line[i][0])] = color_1
    if circle_1 is not None:
        for i in range(circle_1.shape[0]):
            if int(circle_1[i][0]) < bscan.shape[1]-1 and int(circle_1[i][1]) < bscan.shape[0]-2:
                bscan_color[int(circle_1[i][1]), int(circle_1[i][0])] = color_4
                # bscan_color[int(circle_1[i][1]), int(circle_1[i][0]) + 1] = color_4
                # bscan_color[int(circle_1[i][1]), int(circle_1[i][0]) - 1] = color_4
                # bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0])] = color_4
                # bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0])] = color_4
                # bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0]) + 1] = color_4
                # bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0]) - 1] = color_4
                # bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0]) - 1] = color_4
                # bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0]) + 1] = color_4
    if circle_2 is not None:
        for i in range(circle_2.shape[0]):
            if int(circle_2[i][0]) < bscan.shape[1]-1 and int(circle_2[i][1]) < bscan.shape[0]-2:
                bscan_color[int(circle_2[i][1]), int(circle_2[i][0])] = color_2
                # bscan_color[int(circle_2[i][1]), int(circle_2[i][0]) + 1] = color_2
                # bscan_color[int(circle_2[i][1]), int(circle_2[i][0]) - 1] = color_2
                # bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0])] = color_2
                # bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0])] = color_2
                # bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0]) + 1] = color_2
                # bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0]) - 1] = color_2
                # bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0]) - 1] = color_2
                # bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0]) + 1] = color_2
    if circle_3 is not None:
        for i in range(circle_3.shape[0]):
            if int(circle_3[i][0]) < bscan.shape[1]-1 and int(circle_2[i][1]) < bscan.shape[0]-1:
                bscan_color[int(circle_3[i][1]), int(circle_3[i][0])] = color_3
                # bscan_color[int(circle_3[i][1]), int(circle_3[i][0]) + 1] = color_3
                # bscan_color[int(circle_3[i][1]), int(circle_3[i][0]) - 1] = color_3
                # bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0])] = color_3
                # bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0])] = color_3
                # bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0]) + 1] = color_3
                # bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0]) - 1] = color_3
                # bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0]) - 1] = color_3
                # bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0]) + 1] = color_3
    os.makedirs("../../data/suture_experiment/painted_png_data/" + serial_num + "/", exist_ok=True)
    cv.imwrite("../../data/suture_experiment/painted_png_data/" +
               serial_num + '/' + str(idx) + "_painted.png", bscan_color)


def gen_path(center, c_idx, start_Z, tip_rad, rot_angle, pix_x, pix_h, pix_z, serial_num):
    circle_tr_in = np.array(draw_part_circle_3d(tip_rad - 0.340, center, start_Z, rot_angle, pix_x, pix_h, pix_z))
    circle_tr_ce = np.array(draw_part_circle_3d(tip_rad, center, start_Z, rot_angle, pix_x, pix_h, pix_z))
    circle_tr_ou = np.array(draw_part_circle_3d(tip_rad + 0.107, center, start_Z, rot_angle, pix_x, pix_h, pix_z))
    line = np.array(pd.read_csv("../../data/suture_experiment/segmentation_line_files/" + serial_num + "_line.csv",
                                header=None))
    bscan = cv.imread('../../data/suture_experiment/png_data/' +
                       serial_num + '/' + str(c_idx) + '_bscan.png', cv.IMREAD_GRAYSCALE)
    paint_bscan(bscan, serial_num,
                idx=c_idx, circle_1=circle_tr_ce, circle_2=circle_tr_in, circle_3=circle_tr_ou, line=line)


if __name__ == "__main__":
    serial_num = '200229L'
    dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                   serial_num + "/dimension.csv", header=None))
    center_coordinate = np.array(pd.read_csv("../../data/suture_experiment/suture_center_files/" +
                                             serial_num + "_suture_center.csv", header=None))
    pix_x = dim_arr[0][0] / dim_arr[1][0]
    pix_h = dim_arr[0][1] / dim_arr[1][1]
    pix_z = dim_arr[0][2] / dim_arr[1][2]
    center = np.array([(center_coordinate[0, 0]/dim_arr[1][0]) * dim_arr[0][0],
                       (center_coordinate[1, 0]/dim_arr[1][1]) * dim_arr[0][1]])
    suture_rad = 4.078
    c_idx = 0
    start_Z = 5.
    rot_angle = 180
    for i in tqdm(range(386, 429)):
        gen_path(center, c_idx + i, start_Z, suture_rad, rot_angle, pix_x, pix_h, pix_z, serial_num)
