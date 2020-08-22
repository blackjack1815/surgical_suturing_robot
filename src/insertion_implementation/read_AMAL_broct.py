import numpy as np
import imageio

import os

from PyBROCT.io.reader import scans


def save_ideal_png(ideal_cscan, serial_num):
    """
    :param ideal_cscan: {int} -- the index of the ideal C-scan
    :param serial_num: {string} -- the serial number of needle insertion
    :return: None
    """
    for data in scans("../../data/suture_experiment/suture_result_broct_files/AMAL_wound/"
                      + serial_num + "_wound.broct"):
        volume = data['volume']
        bscan = volume[ideal_cscan-1]
        bscan = bscan[::-1, :]
        bscan = (bscan - 54) / (100 - 54)
        bscan = np.clip(bscan, 0, 1)
        bscan = (255 * bscan).astype(np.uint8)
        os.makedirs("../../data/suture_experiment/suture_result_broct_files/AMAL_png/" +
                    serial_num + "/", exist_ok=True)
        imageio.imwrite('../../data/suture_experiment/suture_result_broct_files/AMAL_png/' +
                        serial_num + '/bscan.png', bscan, format='png')


if __name__ == "__main__":
    serial_num = "200305O"
    ideal_cscan = 250
    save_ideal_png(ideal_cscan, serial_num)