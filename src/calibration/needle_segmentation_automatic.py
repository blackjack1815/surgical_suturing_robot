import cv2 as cv
from open3d import *
from tqdm import tqdm
import pandas as pd
import os

from PyBROCT.io.reader import scans


def pcd_y_clip(volume, clip_len):
    """
    :param volume: {numpy.array} -- the OCT volume
    :param clip_len: {int} -- the length will be removed in the y direction
    :return: clipped_vol: {numpy.array} -- the OCT volume after the clipping
    """
    clipped_vol = volume[:, clip_len[0]:clip_len[1], :]
    return clipped_vol


class AutomaticSegmentation:
    def __init__(self, serial_num, threshold, num, map_thres, clip_threshold_xz, clip_threshold_y):
        """
        :param serial_num: {string} -- the serial number of calibration data
        :param threshold: {int} -- the threshold of the brightness of pixel
        :param num: {int} -- the number of the volume
        :param map_thres: {int} -- the pixel threshold to generate binary map
        :param clip_threshold_xz: {list} -- the clipping threshold on xz
        :param clip_threshold_y: {list} -- the clipping threshold on y
        """
        self.threshold = threshold
        self.map_thres = map_thres
        self.num = num
        self.serial_num = serial_num
        self.dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                            self.serial_num + "/dimension.csv", header=None))
        self.clip_threshold_xz = clip_threshold_xz
        self.clip_threshold_y = clip_threshold_y

    def intensity_projection(self, index, brightest_pix):
        """
        :param index: {int} -- the index of the .broct file
        :param brightest_pix: {int} -- the brightest pixel value
        :return: thres_map: {numpy.array} -- the intensity projection threshold map
        :return: volume: {numpy.array} -- the OCT volume
        :return: brightest_pix: {int} -- the brightest pixel value
        """
        for data in scans("../../data/oct_volume_calibration/" +
                          self.serial_num + "/config" + str(index) + ".broct"):
            volume = data['volume']
            for i in tqdm(range(volume.shape[0])):
                volume[i] = volume[i][::-1, :]
            volume_copy = np.copy(volume)
            volume = pcd_y_clip(volume, self.clip_threshold_y[index-1])
            volume_sum = np.max(volume, axis=1)
            volume_sum -= volume_sum.min()
            if index == 1:
                brightest_pix = volume_sum.max()
            volume_sum = np.array(volume_sum * (100 / (brightest_pix - volume_sum.min())), dtype=np.uint8)

            # ### DEBUG ###
            cv.namedWindow('component', cv.WINDOW_NORMAL)
            cv.imshow('component', volume_sum)
            cv.waitKey(10)

            thres_map = cv.threshold(volume_sum, self.map_thres, 255, cv.THRESH_BINARY)[1]
            if index == 6:
                thres_map = cv.morphologyEx(thres_map, cv.MORPH_CLOSE, np.ones((3, 3)))
            else:
                thres_map = cv.morphologyEx(thres_map, cv.MORPH_CLOSE, np.ones((5, 5)))

            if self.clip_threshold_xz[index-1][1] == -1:
                thres_map[:, :self.clip_threshold_xz[index-1][0]] = 0
            else:
                thres_map[:, self.clip_threshold_xz[index-1][0]:] = 0
            if self.clip_threshold_xz[index-1][2] != 0:
                thres_map[self.clip_threshold_xz[index-1][2]:, :] = 0

            volume_copy[:, :self.clip_threshold_y[index-1][0] + 1, :] = 0
            return thres_map, volume_copy, brightest_pix

    def connected_components(self, thres_map, volume, index):
        """
        :param thres_map: {numpy.array} -- the intensity projection threshold map
        :param volume: {numpy.array} -- the OCT volume
        :param index: {int} -- the index of the .broct file
        :return: None
        """
        ret, labels = cv.connectedComponents(thres_map)
        _, counts = np.unique(labels, return_counts=True)
        mask = np.zeros(labels.shape, dtype=np.uint8)
        max_index = np.argmax(counts[1:])
        mask[labels == max_index + 1] = 255

        # ### DEBUG ###
        cv.namedWindow('mask', cv.WINDOW_NORMAL)
        cv.imshow('mask', mask)
        cv.waitKey(10)

        volume_clean = np.zeros(volume.shape)
        volume_clean[np.where(mask == 255)[0], :, np.where(mask == 255)[1]] = \
            volume[np.where(mask == 255)[0], :, np.where(mask == 255)[1]]
        pcd = self.pointcloud_process(volume_clean, volume)
        cl, ind = statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)
        pcd = select_down_sample(pcd, ind)
        cl, ind = statistical_outlier_removal(pcd, nb_neighbors=15, std_ratio=2.0)
        pcd = select_down_sample(pcd, ind)
        mesh_frame = create_mesh_coordinate_frame(size=0.6, origin=[0, 0, 0])
        draw_geometries([pcd, mesh_frame])
        os.makedirs("../../data/point_clouds/source_point_clouds/" + self.serial_num + "/", exist_ok=True)
        write_point_cloud("../../data/point_clouds/source_point_clouds/" + self.serial_num +
                          "/cropped_OCT_config_" + str(index) + ".ply", pcd)

    def pointcloud_process(self, volume_clean, volume_dirty):
        """
        :param volume_clean: {PyBROCT.data} -- the data of the clean volume of the Bscans
        :param volume_dirty: {PyBROCT.data} -- the data of the dirty volume of the Bscans
        :return: pcd: {open3d.PointCloud} -- the point cloud file
        """
        x_unit = self.dim_arr[0][0] / self.dim_arr[1][0]
        y_unit = self.dim_arr[0][1] / self.dim_arr[1][1]
        z_unit = self.dim_arr[0][2] / self.dim_arr[1][2]

        max_Ascan_index = np.argmax(volume_clean, axis=1)
        pointcloud_singlelay = []
        specular_reflection = []
        for i in tqdm(range(max_Ascan_index.shape[0])):
            for j in range(max_Ascan_index.shape[1]):
                if volume_clean[i, max_Ascan_index[i][j], j] >= self.threshold:
                    pointcloud_singlelay.append([i, max_Ascan_index[i][j], j])
                    if j >= max_Ascan_index.shape[1] - 8 or j <= 15:
                        specular_reflection.append([i, max_Ascan_index[i][j]])
        max_Ascan_index = np.argmax(volume_dirty, axis=1)
        for i in tqdm(range(max_Ascan_index.shape[0])):
            for j in range(max_Ascan_index.shape[1]):
                if volume_dirty[i, max_Ascan_index[i][j], j] >= self.threshold and\
                        (j >= max_Ascan_index.shape[1] - 8 or j <= 15):
                    specular_reflection.append([i, max_Ascan_index[i][j]])
        pointcloud_singlelay_clean = []
        for i in pointcloud_singlelay:
            remove_sign = False
            for j in specular_reflection:
                if j[0] - 5 <= i[0] <= j[0] + 5 and j[1] - 3 <= i[1] <= j[1] + 3:
                    remove_sign = True
                    break
            if remove_sign is False:
                pointcloud_singlelay_clean.append(i)
        pointcloud = np.multiply(pointcloud_singlelay_clean, np.array([z_unit, y_unit, x_unit]))
        pointcloud = np.flip(pointcloud, 1)
        pcd = PointCloud()
        pcd.points = Vector3dVector(pointcloud)
        return pcd

    def segment_all_volume(self):
        """
        :return: None
        """
        brightest_pix = 59
        for i in range(1, self.num + 1):
            print("Generating " + str(i) + "th point cloud")
            thres_map, volume_copy, brightest_pix = self.intensity_projection(i, brightest_pix)
            self.connected_components(thres_map, volume_copy, i)


if __name__ == '__main__':
    clip_threshold_xz = [[430, -1, 0],
                         [230, -1, 0],
                         [0, -1, 0],
                         [430, -1, 0],
                         [220, -1, 0],
                         [0, -1, 0],
                         [715, 1, 0],
                         [550, 1, 0],
                         [350, 1, 300]]
    clip_threshold_y = [[550, -1],
                        [550, -1],
                        [550, -1],
                        [550, -90],
                        [550, -90],
                        [550, -90],
                        [550, -90],
                        [550, -90],
                        [550, -90]]

    #230

    # clip_threshold_xz = [[250, -1, 0],
    #                      [50, -1, 0],
    #                      [550, 1, 0]]
    # clip_threshold_y = [[500, -1],
    #                     [500, -1],
    #                     [500, -90]]

    serial_num = "190911A"
    threshold = 75
    map_threhold = 11
    num = 9
    segmentation = AutomaticSegmentation(serial_num, threshold, num, map_threhold, clip_threshold_xz, clip_threshold_y)
    segmentation.segment_all_volume()
