import struct
import socket
import asyncio

import websockets
import numpy as np
import cv2 as cv
import pandas as pd
from klampt.math import so3, se3


# This function only generate suture path on the direction of Bscan
def draw_part_circle_3d(radius, center, start_Z, rot_angle, start_angle, rot_x, pix_x, pix_h, pix_z):
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
    rot_mat = (so3.from_rpy([rot_x, 0.0, 0.0]), [0.0]*3)
    for i in range(0, rot_angle*10):
        current_angle = start_angle + i*0.1
        pos_x_0 = center[0] - radius * np.cos((current_angle * np.pi) / 180.)
        pos_y_0 = center[1] + radius * np.sin((current_angle * np.pi) / 180.)
        milestone = np.divide(se3.apply(rot_mat, [pos_x_0, pos_y_0, start_Z]), [pix_x, pix_h, pix_z])
        suture_path.append(milestone)
    return suture_path


def process(volume):
    serial_num = '200229L'
    center_coordinate = np.array(pd.read_csv("../../data/suture_experiment/suture_center_files/" +
                                             serial_num + "_suture_center.csv", header=None))
    dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                   serial_num + "/dimension.csv", header=None))
    idx = 24
    pix_x = dim_arr[0][0] / dim_arr[1][0]
    pix_h = dim_arr[0][1] / dim_arr[1][1]
    pix_z = dim_arr[0][2] / dim_arr[1][2]
    center = np.array([(center_coordinate[0, 0]/dim_arr[1][0]) * dim_arr[0][0],
                       (center_coordinate[1, 0]/dim_arr[1][1]) * dim_arr[0][1]])
    center_rad = 3.961
    rot_x_val = -0.0
    circle_tr_in = draw_part_circle_3d(center_rad-0.223, center, 5., 180, 0, rot_x_val, pix_x, pix_h, pix_z)
    circle_tr_in = np.array(circle_tr_in)
    circle_tr_ce = draw_part_circle_3d(center_rad+0.244, center, 5., 180, 0, rot_x_val, pix_x, pix_h, pix_z)
    circle_tr_ce = np.array(circle_tr_ce)
    circle_tr_ou = draw_part_circle_3d(center_rad+0.224, center, 5., 180, 0., rot_x_val, pix_x, pix_h, pix_z)
    circle_tr_ou = np.array(circle_tr_ou)
    line = np.array(pd.read_csv("../../data/suture_experiment/segmentation_line_files/" + serial_num + "_line.csv",
                                header=None))
    show_bscan(volume[::-1, :, idx], name='B-scan {}'.format(idx),
               circle_1=circle_tr_in, circle_2=circle_tr_ce, circle_3=circle_tr_ou, line=line)
    if cv.waitKey(1) == 27:
        raise SystemExit()


def show_bscan(bscan, black=50, white=90, name=None,
               circle_1=None, circle_2=None, circle_3=None, line=None):
    bscan = bscan.astype(np.float32)
    bscan = (bscan - black) / (white - black)
    bscan = 255 * np.clip(bscan, 0, 1)
    bscan_color = cv.cvtColor(bscan.astype(np.uint8), cv.COLOR_GRAY2BGR)

    bscan_color = cv.flip(bscan_color, 1)
    color_1 = [0, 0, 255]
    color_2 = [0, 255, 0]
    color_3 = [255, 0, 0]
    if line is not None:
        for i in range(line.shape[0]):
            if int(line[i][0]) <= 715:
                bscan_color[int(line[i][1]), int(line[i][0])] = [0, 255, 200]
                bscan_color[int(line[i][1]+1), int(line[i][0])] = [0, 255, 200]
                bscan_color[int(line[i][1]-1), int(line[i][0])] = [0, 255, 200]
    if circle_1 is not None:
        for i in range(circle_1.shape[0]):
            if int(circle_1[i][0]) < bscan.shape[1]-1 and int(circle_1[i][1]) < bscan.shape[0]-2:
                    # and (idx*pix_z - pix_z/2) < circle_1[i][2]*pix_z < (idx*pix_z + pix_z/2):
                bscan_color[int(circle_1[i][1]), int(circle_1[i][0])] = color_1
                bscan_color[int(circle_1[i][1]), int(circle_1[i][0]) + 1] = color_1
                bscan_color[int(circle_1[i][1]), int(circle_1[i][0]) - 1] = color_1
                bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0])] = color_1
                bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0])] = color_1
                bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0]) + 1] = color_1
                bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0]) - 1] = color_1
                bscan_color[int(circle_1[i][1] + 1), int(circle_1[i][0]) - 1] = color_1
                bscan_color[int(circle_1[i][1] - 1), int(circle_1[i][0]) + 1] = color_1
    if circle_2 is not None:
        for i in range(circle_2.shape[0]):
            if int(circle_2[i][0]) < bscan.shape[1]-1 and int(circle_2[i][1]) < bscan.shape[0]-2:
                     # and (idx*pix_z - pix_z/2) < circle_2[i][2]*pix_z < (idx*pix_z + pix_z/2):
                bscan_color[int(circle_2[i][1]), int(circle_2[i][0])] = color_2
                bscan_color[int(circle_2[i][1]), int(circle_2[i][0]) + 1] = color_2
                bscan_color[int(circle_2[i][1]), int(circle_2[i][0]) - 1] = color_2
                bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0])] = color_2
                bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0])] = color_2
                bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0]) + 1] = color_2
                bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0]) - 1] = color_2
                bscan_color[int(circle_2[i][1] + 1), int(circle_2[i][0]) - 1] = color_2
                bscan_color[int(circle_2[i][1] - 1), int(circle_2[i][0]) + 1] = color_2
    if circle_3 is not None:
        for i in range(circle_3.shape[0]):
            if int(circle_3[i][0]) < bscan.shape[1]-1 and int(circle_2[i][1]) < bscan.shape[0]-1:
                    # and (idx*pix_z - pix_z/2) < circle_3[i][2]*pix_z < (idx*pix_z + pix_z/2):
                bscan_color[int(circle_3[i][1]), int(circle_3[i][0])] = color_3
                bscan_color[int(circle_3[i][1]), int(circle_3[i][0]) + 1] = color_3
                bscan_color[int(circle_3[i][1]), int(circle_3[i][0]) - 1] = color_3
                bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0])] = color_3
                bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0])] = color_3
                bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0]) + 1] = color_3
                bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0]) - 1] = color_3
                bscan_color[int(circle_3[i][1] + 1), int(circle_3[i][0]) - 1] = color_3
                bscan_color[int(circle_3[i][1] - 1), int(circle_3[i][0]) + 1] = color_3
    cv.namedWindow(name or 'Scan', cv.WINDOW_NORMAL)
    cv.imshow(name or 'Scan', bscan_color)


@asyncio.coroutine
def run_socket(oct_address, oct_port, **kwargs):
    address = kwargs.pop('address', None)
    port = kwargs.pop('port', None)
    if address:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((address, port))
    else:
        sock = None

    ws = yield from websockets.connect('ws://{}:{}'.format(oct_address, oct_port), max_size=2**30)

    # subscribe to volume data at display rate
    data = struct.pack('!ii', 6, 5)
    yield from ws.send(data)

    volume_chunks = []

    while True:
        data = yield from ws.recv()

        header_fmt = '>iiiii'
        n = struct.calcsize(header_fmt)

        (msg_type, xdim, ydim, zdim, offset) = struct.unpack(header_fmt, data[:n])

        if not volume_chunks and offset > 0:
            # wait for first chunk
            continue

        if volume_chunks and offset == 0:
            # assemble the volume
            volume = np.dstack(volume_chunks)
            volume_chunks.clear()

            # process the volume
            process(volume, **kwargs)

        chunk = np.fromstring(data[n:], dtype=np.uint8, count=xdim * ydim * zdim).reshape((zdim, ydim, xdim))
        chunk = np.swapaxes(chunk, 0, 2)
        chunk = np.swapaxes(chunk, 0, 1)
        volume_chunks.append(chunk)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='OCT network volume parser', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--address', '-a', type=str, help='network address for OCT', default='192.168.1.178')
    parser.add_argument('--port', '-p', type=int, help='network port for OCT', default=1234)

    args = parser.parse_args()

    kwargs = dict()

    asyncio.get_event_loop().run_until_complete(run_socket(args.address, args.port, **kwargs))
