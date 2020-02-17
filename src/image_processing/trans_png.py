import cv2 as cv
import numpy
import imageio

from PyBROCT.io.reader import scans


if __name__ == '__main__':
    serial_num = '190911A'
    for data in scans("../../data/suture_experiment/suture_result_broct_files/suture_res_" + serial_num + ".broct"):
        volume = data['volume']
        idx = 1
        for bscan in volume:
            bscan = bscan[::-1, :]
            # white and black levels
            bscan = (bscan - 54) / (100 - 54)
            bscan = numpy.clip(bscan, 0, 1)
            bscan = (255 * bscan).astype(numpy.uint8)
            print("The Bscan is: ", bscan.shape)
            rows, cols = map(int, bscan.shape)
            cv.namedWindow('B-Scan', cv.WINDOW_NORMAL)
            cv.imshow("B-Scan", bscan)
            imageio.imwrite('../../data/suture_experiment/png_data/' +
                            serial_num + '/' + str(idx) + '_bscan.png', bscan, format='png')
            k = chr(cv.waitKey(-1))
            if k == 'q':
                raise SystemExit
