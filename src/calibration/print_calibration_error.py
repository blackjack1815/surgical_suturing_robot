import numpy as np


def print_error(serial_num):
    """

    :param serial_num: the serial number of the calibration file
    :return: None
    """
    print("Frame: ", np.load("../../data/suture_experiment/calibration_result_files/"
                             + serial_num + "/calibration_result.npy"))
    print("RMSE: ", np.load("../../data/suture_experiment/calibration_result_files/"
                            + serial_num + "/calibration_error_rmse.npy"))
    print("Error Lists: ", np.load("../../data/suture_experiment/calibration_result_files/"
                                   + serial_num + "/calibration_error_list.npy"))
    print("Error Matrices: ", np.load("../../data/suture_experiment/calibration_result_files/"
                                      + serial_num + "/calibration_error_mat.npy"))
    print("Calibration Result: ", np.load("../../data/suture_experiment/calibration_result_files/"
                                          + serial_num + "/calibration_result.npy"))


if __name__ == "__main__":
    serial_num = "200228B"
    print_error(serial_num)
