import automatic_sampling_s0 as s0
import automatic_sampling_s1 as s1
import automatic_sampling_s2 as s2
import calibration as cal
import needle_segmentation_automatic as seg


def main():
    serial_num = "200305O"
    controller = s0.RecordController(serial_num, '192.168.1.178')
    controller.start()

    st_config_name = "../../data/robot_configuration_files/configuration_record_200224Y.csv"
    es_config_name = "../../data/robot_configuration_files/configuration_record_" + serial_num + "A.csv"
    s1.sampling_3(st_config_name, es_config_name, serial_num)

    clip_threshold_xz = [[250, -1, 0],
                         [50, -1, 0],
                         [550, 1, 0]]
    clip_threshold_y = [[550, -1],
                        [550, -1],
                        [550, -90]]
    threshold = 75
    map_threhold = 11
    num = 3
    segmentation = seg.AutomaticSegmentation(serial_num + "B",
                                             threshold, num, map_threhold, clip_threshold_xz, clip_threshold_y)
    segmentation.segment_all_volume()

    robot_filename = "../../data/robot_model_files"
    num_pcd = 3
    calibration = cal.Calibration(robot_filename, serial_num + 'B', num_pcd)
    calibration.run_calibration(2, 1)

    s2.sampling_9(serial_num)

    clip_threshold_xz = [[430, -1, 0],
                         [220, -1, 0],
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
    num = 9
    segmentation = seg.AutomaticSegmentation(serial_num, threshold
                                             , num, map_threhold, clip_threshold_xz, clip_threshold_y)
    segmentation.segment_all_volume()

    num_pcd = 9
    calibration = cal.Calibration(robot_filename, serial_num, num_pcd)
    calibration.run_calibration(6, 5)


if __name__ == "__main__":
    main()
