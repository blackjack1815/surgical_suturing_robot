from open3d import *
from tqdm import tqdm
import pandas as pd

from PyBROCT.io.reader import scans


class ManualSegmentation:
    def __init__(self, serial_num, threshold, num):
        """

        :param serial_num: {string} -- the serial number of calibration data
        :param threshold: {int} -- the threshold of the brightness of pixel
        :param num: {int} -- the number of the volume
        :param root_filename: {string} -- the file location of the broct file
        """
        self.threshold = threshold
        self.num = num
        self.serial_num = serial_num
        self.dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                            self.serial_num + "/dimension.csv", header=None))

    def manual_segmentation(self, index):
        """

        :param index: {int} -- the index of the .broct file
        :return: None
        """
        for (index, data) in scans("../../data/oct_volume_calibration/" +
                                   self.serial_num + "/config" + str(index) + ".broct"):
            volume = data['volume']
            # volume = volume[:, :, 85:]
            volume = volume[:, :, :]
            volume_raw = np.array(volume, copy=True)
            for i in tqdm(range(volume.shape[0])):
                volume_raw[i] = volume_raw[i][::-1, :]
            pcd_s = self.pointcloud_process(volume_raw)
            pcd_s.paint_uniform_color([1, 0.706, 0])
            draw_geometries_with_editing([pcd_s])

    def pointcloud_process(self, volume):
        """

        :param volume: {PyBROCT.data} -- the data of the volume of the Bscans
        :return: pcd: {open3d.PointCloud} -- the point cloud file
        """
        x_unit = self.dim_arr[0][0] / self.dim_arr[1][0]
        y_unit = self.dim_arr[0][1] / self.dim_arr[1][1]
        z_unit = self.dim_arr[0][2] / self.dim_arr[1][2]

        max_Ascan_index = np.argmax(volume, axis=1)
        pointcloud_singlelay = []
        for i in tqdm(range(max_Ascan_index.shape[0])):
            for j in range(max_Ascan_index.shape[1]):
                if volume[i, max_Ascan_index[i][j], j] >= self.threshold:
                    pointcloud_singlelay.append([i, max_Ascan_index[i][j], j])
        pointcloud = np.multiply(pointcloud_singlelay, np.array([z_unit, y_unit, x_unit]))
        pointcloud = np.flip(pointcloud, 1)
        pcd = PointCloud()
        pcd.points = Vector3dVector(pointcloud)
        return pcd

    def segment_all_volume(self):
        """

        :return: None
        """
        for i in range(1, self.num + 1):
            print("Generating " + str(i) + "th point cloud")
            self.manual_segmentation(i)


if __name__ == '__main__':
    serial_num = "190911A"
    threshold = 65
    num = 9
    segmentation = ManualSegmentation(serial_num, threshold, num)
    segmentation.segment_all_volume()
