import numpy
import imageio

import os

from PyBROCT.io.reader import scans


if __name__ == '__main__':
    serial_num = '200305O'
    for data in scans("../../data/suture_experiment/suture_result_broct_files/suture/" + serial_num + "_suture.broct"):
        volume = data['volume']
        idx = 222
        for bscan in volume[222:256]:
            bscan = bscan[::-1, :]
            # white and black levels
            bscan = (bscan - 54) / (100 - 54)
            bscan = numpy.clip(bscan, 0, 1)
            bscan = (255 * bscan).astype(numpy.uint8)
            os.makedirs("../../data/suture_experiment/png_data/" + serial_num + "/", exist_ok=True)
            imageio.imwrite('../../data/suture_experiment/png_data/' +
                            serial_num + '/' + str(idx) + '_bscan.png', bscan, format='png')
            idx += 1
