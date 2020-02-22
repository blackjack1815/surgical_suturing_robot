from klampt import *
from klampt import vis
from klampt.math import se3, so3, vectorops
from open3d import *
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.optimize import root

import copy
import time
import random


def redraw_pcd(pcd_pair, Toct_needle):
    """

    :param pcd_pair: {list} -- The list of Open3D.PointCloud
    :param Toct_needle: {list} -- The transfer matrix from OCT frame to needle frame
    :return: None
    """
    source = pcd_pair[0]
    target = pcd_pair[1]
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(se3.homogeneous(Toct_needle))
    return [source_temp, target_temp]


def trans_all2needle(Tlink_set, Toct, Tneedle, pcd_list):
    """

    :param Tlink_set: {list} -- the robot configuration list
    :param Toct: {tuple} -- the transformation matrix from world to OCT
    :param Tneedle: {tuple} -- the transformation matrix from robot end effector to needle
    :param pcd_list: {list} -- the point cloud list
    :return: None
    """
    pcd_needle_list = []
    pcd_t = copy.deepcopy(pcd_list[6][1])
    pcd_t.paint_uniform_color([0, 0.651, 0.929])
    pcd_needle_list.append(pcd_t)
    mesh_frame = create_mesh_coordinate_frame(0.0006, [0, 0, 0])
    pcd_needle_list.append(mesh_frame)
    for i in range(len(Tlink_set)):
        Tneedle2oct = se3.mul(se3.inv(Tneedle), se3.mul(se3.inv(Tlink_set[i]), Toct))
        pcd_s_copy = copy.deepcopy(pcd_list[i][0])
        pcd_s_copy.transform(se3.homogeneous(Tneedle2oct))
        pcd_s_copy.paint_uniform_color([0.4 + i * 0.1, 0.706, 0])
        pcd_needle_list.append(pcd_s_copy)
    draw_geometries(pcd_needle_list)


def draw_registration_result(source, target, transformation):
    """

    :param source: {Open3D.PointCloud} -- the source point cloud
    :param target: {Open3D.PointCloud} -- the target point cloud
    :param transformation: {list} -- the transformation matrix from the target point cloud to source point cloud
    :return: None
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(np.linalg.inv(transformation))
    mesh_frame = create_mesh_coordinate_frame(size=0.0006, origin=[0, 0, 0])
    draw_geometries([source_temp, target_temp, mesh_frame])


def duplicate_part_pcd(trans, pcd, x_thre, y_thre, ratio):
    """

    :param trans: {numpy.array} -- the transfer matrix from source point cloud to target point cloud
    :param pcd: {open3d.PointCloud} -- the source point cloud
    :param x_thre: {float} -- the threshold of x value to take the duplication
    :param y_thre: {float} -- the threshold of y value to take the duplication
    :param ratio: {int} -- the ratio to duplicate the points
    :return: duplicated_pcd: {open3d.PointCloud} -- the source point cloud after duplicating
    """
    pcd.transform(trans)
    points = np.asarray(pcd.points)
    duplicated_points = np.copy(points)
    for i in tqdm(range(points.shape[0])):
        if points[i][0] > x_thre and points[i][1] > y_thre:
            duplicated_points = np.append(duplicated_points, np.repeat(np.array([points[i]]), ratio, axis=0), axis=0)
    duplicated_pcd = PointCloud()
    duplicated_pcd.points = Vector3dVector(duplicated_points)
    duplicated_pcd.transform(np.linalg.inv(trans))
    duplicated_pcd.paint_uniform_color([1, 0.0, 0.929])
    print("The raw point cloud shape is ", points.shape,
          "\nThe duplicated point cloud shape is ", duplicated_points.shape)
    pcd.transform(np.linalg.inv(trans))
    return duplicated_pcd


def opt_error_fun(est_input, *args):
    """

    :param est_input: {numpy.array} -- the initial guess of the transformation of Toct and Tneedle
    :param args: {tuple} -- the set of tranfermation of Tlink and ICP transformation matrix
    :return: error: {float} -- the error between the true transformation and estimated transformation
    """
    Roct = so3.from_rpy(est_input[0:3])
    toct = est_input[3:6]
    Rn = so3.from_rpy(est_input[6:9])
    tn = est_input[9:12]
    Tlink_set = args[0]
    Tneedle2oct_icp = args[1]
    fun_list = np.array([])
    for i in range(len(Tlink_set)):
        fun_list = np.append(fun_list, np.multiply(so3.error(so3.inv(Tneedle2oct_icp[i][0]),
                                                             so3.mul(so3.mul(so3.inv(Roct), Tlink_set[i][0]), Rn)), 1))
        fun_list = np.append(fun_list,
                             np.multiply(vectorops.sub(vectorops.sub([0., 0., 0.],
                                                                     so3.apply(so3.inv(Tneedle2oct_icp[i][0]),
                                                                               Tneedle2oct_icp[i][1])),
                                                       so3.apply(so3.inv(Roct),
                                                                 vectorops.add(vectorops.sub(Tlink_set[i][1], toct),
                                                                               so3.apply(Tlink_set[i][0], tn)))), 1000))

    return fun_list


def opt_error_fun_penalize(est_input, *args):
    """

    :param est_input: {numpy.array} -- the initial guess of the transformation of Toct and Tneedle
    :param args: {tuple} -- the set of tranfermation of Tlink and ICP transformation matrix
    :return: error: {float} -- the error between the true transformation and estimated transformation
    """
    est_Toct = (so3.from_rpy(est_input[0:3]), est_input[3:6])
    est_Tneedle = (so3.from_rpy(est_input[6:9]), est_input[9:12])
    Tlink_set = args[0]
    Tneedle2oct_icp = args[1]
    fun_list = np.array([])
    for i in range(len(Tlink_set)):
        Toct2needle_est = se3.mul(se3.mul(se3.inv(est_Toct), Tlink_set[i]), est_Tneedle)
        fun_list = np.append(fun_list, np.absolute(np.multiply(se3.error(Tneedle2oct_icp[i],
                                                             se3.inv(Toct2needle_est))[0:3], [1, args[2], 1])))
        fun_list = np.append(fun_list, np.absolute(np.multiply(se3.error(Tneedle2oct_icp[i],
                                                             se3.inv(Toct2needle_est))[3:6], 1000)))

    return fun_list


def get_world(world_file, robot_file, visualize_robot=True):
    """

    :param world_file: {string} -- the location of the world file
    :param robot_file: {string} -- the location of robot file
    :param visualize_robot: {bool} -- True-visualize robot False-do not visualize robot
    :return: world: {Klampt.WorldModel} -- the model of the world in Klmapt
    """
    world = WorldModel()
    res = world.readFile(world_file)
    if not res:
        raise RuntimeError("Unable to load terrain model")
    if visualize_robot:
        res = world.readFile(robot_file)
        if not res:
            raise RuntimeError("Unable to load robot model")
    return world


class Calibration:
    def __init__(self, robot_filename, serial_num, num_pcd):
        """

        :param robot_filename: {string} -- the location of the Klampt's robot files
        :param serial_num: {string} -- the serial number of the calibration
        :param num_pcd: {int} -- the number of needle point clouds (poses)
        :return: None
        """
        self.world_file = robot_filename + "/worlds/flatworld.xml"
        self.robot_file = robot_filename + "/robots/irb120_icp.rob"
        self.world = get_world(self.world_file, self.robot_file, visualize_robot=True)
        self.robot = self.world.robot(0)
        self.config_filename = "../../data/robot_configuration_files/configuration_record_" + serial_num + ".csv"
        self.serial_num = serial_num
        self.qstart = self.robot.getConfig()
        self.num_pcd = num_pcd
        self.Clink_set = []
        self.Tlink_set = []
        self.trans_list = []
        self.pcd_list = []
        self.min_res = None
        vis.add("world", self.world)

    def sample_config(self):
        """

        :return: None
        """
        self.Clink_set = np.array(pd.read_csv(self.config_filename, header=None)).tolist()
        for i in range(len(self.Clink_set)):
            self.Clink_set[i] = np.multiply(self.Clink_set[i], np.pi / 180.).tolist()
            self.Clink_set[i] = [0] * 7 + self.Clink_set[i] + [0]
        for i in range(len(self.Clink_set)):
            vis.add("configure " + str(i), self.Clink_set[i])
            vis.setColor("configure " + str(i), i * 0.15, 1, i * 0.05, 0.5)
            self.robot.setConfig(self.Clink_set[i])

    def link_transform(self):
        """

        :return: None
        """
        for i in range(len(self.Clink_set)):
            self.robot.setConfig(self.Clink_set[i])
            Trans2World = self.robot.link('link_6').getTransform()
            self.Tlink_set.append(Trans2World)

    def needle2oct_icp(self):
        """

        :return: None
        """
        threshold = 50
        for i in range(0, self.num_pcd):
            pcd_s = read_point_cloud("../../data/point_clouds/source_point_clouds/" + self.serial_num +
                                     "/cropped_OCT_config_" + str(i + 1) + ".ply")
            pcd_s.points = Vector3dVector(np.multiply(np.asarray(pcd_s.points), 1e-3))
            if i < 6:
                pcd_t = read_point_cloud("../../data/point_clouds/target_point_clouds/TF-1-cut.ply")
            else:
                pcd_t = read_point_cloud("../../data/point_clouds/target_point_clouds/TF-1.ply")
            pcd_t.points = Vector3dVector(np.multiply(np.asarray(pcd_t.points), 1e-3))
            temp_rmse = 1.
            count = 0
            while temp_rmse > 2.0e-5:
                print("ICP implementation is in " + str(i+1) + "th point clouds.")
                random.seed(count * 5 + 7 - count / 3)
                if i > 5:
                    trans_init = np.asarray(se3.homogeneous((
                        so3.from_rpy([random.uniform(-0.5 * np.pi, 0.5 * np.pi) for kk in range(3)]),
                        [random.uniform(-5. * 1e-3, 5. * 1e-3) for kk in range(3)])))
                else:
                    trans_init = np.asarray(se3.homogeneous((
                        so3.from_rpy([random.uniform(-0.5 * np.pi, 0.5 * np.pi) + np.pi] +
                                     [random.uniform(-0.5 * np.pi, 0.5 * np.pi) for kk in range(2)]),
                        [random.uniform(-5. * 1e-3, 5. * 1e-3)] + [random.uniform(-5. * 1e-3, 5. * 1e-3) + 15. * 1e-3] +
                        [random.uniform(-5. * 1e-3, 5. * 1e-3)])))
                reg_p2p = registration_icp(pcd_s, pcd_t, threshold, trans_init,
                                           TransformationEstimationPointToPoint(),
                                           ICPConvergenceCriteria(max_iteration=300, relative_rmse=1e-15,
                                                                  relative_fitness=1e-15))
                modified_pcd_s = duplicate_part_pcd(reg_p2p.transformation, pcd_s, -0.2 * 1e-3, 0.0 * 1e-3, 30)
                reg_p2p_2 = registration_icp(modified_pcd_s, pcd_t, threshold, reg_p2p.transformation,
                                             TransformationEstimationPointToPoint(),
                                             ICPConvergenceCriteria(max_iteration=100, relative_rmse=1e-15,
                                                                    relative_fitness=1e-15))
                print("The RMSE is: ", reg_p2p_2.inlier_rmse)
                temp_rmse = reg_p2p_2.inlier_rmse
                count += 1
            trans = se3.from_homogeneous(reg_p2p_2.transformation)
            self.trans_list.append(trans)
            point_oct = []
            pcd_s_copy = PointCloud()
            pcd_p = PointCloud()
            pcd_s_copy.points = Vector3dVector(np.multiply(np.asarray(modified_pcd_s.points), 1))
            pcd_t_copy = read_point_cloud("../../data/point_clouds/target_point_clouds/TF-1.ply")
            pcd_t_copy.points = Vector3dVector(np.multiply(np.asarray(pcd_t_copy.points), 1e-3))
            pcd_p.points = Vector3dVector(point_oct)
            # ***DEBUG*** #
            # draw_registration_result(pcd_s_copy, pcd_t_copy, se3.homogeneous(trans))
            self.pcd_list.append([pcd_s_copy, pcd_t_copy])

    def optimization(self, penalty_y):
        """

        :param penalty_y {float} penalize the rotation y axis during optimization
        :return: None
        """
        time_stamp = time.time()
        min_error = 10e5
        for k in range(25):
            random.seed(time.time())
            est_input = [random.uniform(-2 * np.pi, 2 * np.pi) for i in range(3)] + \
                        [random.uniform(-0.8, 0.8) for i in range(3)] + \
                        [random.uniform(-2 * np.pi, 2 * np.pi) for i in range(3)] + \
                        [random.uniform(-0.2, 0.2) for i in range(3)]
            print("We will start " + str(k + 1) + "th minimize!")
            res = root(opt_error_fun_penalize, np.array(est_input),
                       args=(self.Tlink_set, self.trans_list, penalty_y), method='lm',
                       options={})
            print("The reason for termination: \n", res.message, "\nAnd the number of the iterations are: ", res.nfev)
            error = np.sum(np.absolute(res.fun))
            print("The minimize is end, and the error is: ", error)
            if error <= min_error:
                min_error = error
                self.min_res = res
        print("The optimization uses: ", time.time() - time_stamp, "seconds")
        print("The error is: ", np.sum(np.absolute(self.min_res.fun)))
        print("The optimized T_oct is: ", (self.min_res.x[0:3], self.min_res.x[3:6]))
        print("And the matrix form is: \n",
              np.array(se3.homogeneous((so3.from_rpy(self.min_res.x[0:3]), self.min_res.x[3:6]))))
        print("The optimized T_needle_end is: ", (self.min_res.x[6:9], self.min_res.x[9:12]))
        print("And the matrix form is: \n",
              np.array(se3.homogeneous((so3.from_rpy(self.min_res.x[6:9]), self.min_res.x[9:12]))))
        np.save("../../data/suture_experiment/calibration_result_files/" + self.serial_num +
                "/calibration_result.npy", self.min_res.x, allow_pickle=True)
        vis.spin(float('inf'))

        vis.clear()
        self.robot.setConfig(self.qstart)
        vis.add("world", self.world)
        World_base = model.coordinates.Frame("Base Frame",
                                             worldCoordinates=(so3.from_rpy([0, 0, 0]), [0, 0, 0]))
        vis.add('World_base', World_base)
        est_OCT_coordinate = model.coordinates.Frame("est OCT Frame",
                                                     worldCoordinates=(so3.from_rpy(self.min_res.x[0:3]),
                                                                       self.min_res.x[3:6]))
        vis.add("est OCT Frame", est_OCT_coordinate)
        Tlink_6 = model.coordinates.Frame("Link 6 Frame", worldCoordinates=self.robot.link('link_6').getTransform())
        vis.add('Link 6 Frame', Tlink_6)
        est_Tneedle_world = se3.mul(self.robot.link('link_6').getTransform(),
                                    (so3.from_rpy(self.min_res.x[6:9]), self.min_res.x[9:12]))
        est_needle_coordinate = model.coordinates.Frame("est needle Frame",
                                                        worldCoordinates=est_Tneedle_world)
        vis.add("est needle Frame", est_needle_coordinate)
        vis.spin(float('inf'))

    def error_analysis(self):
        """

        :return: None
        """
        Toct = (so3.from_rpy(self.min_res.x[0:3]), self.min_res.x[3:6])
        Tneedle = (so3.from_rpy(self.min_res.x[6:9]), self.min_res.x[9:12])
        trans_error_list = []
        trans_error_mat = []
        rot_error_list = []
        rot_error_mat = []
        redraw_list = []
        for i in range(len(self.Tlink_set)):
            Toct_needle_est = se3.mul(se3.mul(se3.inv(Toct), self.Tlink_set[i]), Tneedle)
            trans_error = vectorops.sub(se3.inv(self.trans_list[i])[1], Toct_needle_est[1])
            rot_error = so3.error(se3.inv(self.trans_list[i])[0], Toct_needle_est[0])
            trans_error_list.append(vectorops.normSquared(trans_error))
            trans_error_mat.append(np.absolute(trans_error))
            rot_error_list.append(vectorops.normSquared(rot_error))
            rot_error_mat.append(np.absolute(rot_error))
            redraw_list.append(redraw_pcd(self.pcd_list[i], Toct_needle_est)[0])
            redraw_list.append(redraw_pcd(self.pcd_list[i], Toct_needle_est)[1])
        redraw_list.append(create_mesh_coordinate_frame(size=0.0015, origin=[0, 0, 0]))
        draw_geometries(redraw_list)
        trans_error_list = np.array(trans_error_list)
        trans_error_mat = np.array(trans_error_mat)
        rot_error_list = np.array(rot_error_list)
        rot_error_mat = np.array(rot_error_mat)
        rmse_trans = np.sqrt(np.mean(trans_error_list))
        rmse_rot = np.sqrt(np.mean(rot_error_list))
        print("The squared translation error list is:\n ", np.sqrt(trans_error_list), "\nAnd the its RMSE is ",
              rmse_trans)
        print("The mean error in XYZ directions is: ", np.mean(trans_error_mat, axis=0))
        print("The squared rotation error list is:\n ", np.sqrt(rot_error_list), "\nAnd the its RMSE is ", rmse_rot)
        print("The mean error in three rotation vectors is: ", np.mean(rot_error_mat, axis=0))
        np.save("../../data/suture_experiment/calibration_result_files/" + self.serial_num +
                "/calibration_error.npy", np.array([rmse_trans, rmse_rot]), allow_pickle=True)
        trans_all2needle(self.Tlink_set, Toct, Tneedle, self.pcd_list)

    def run_calibration(self, penalty_y):
        """

        :param penalty_y {float} penalize the rotation y axis during optimization
        :return: None
        """
        self.sample_config()
        self.link_transform()
        self.needle2oct_icp()
        self.optimization(penalty_y)
        self.error_analysis()


if __name__ == "__main__":
    robot_filename = "../../data/robot_model_files"
    serial_num = "190911A"
    penaly_rotation_y = 0.1
    num_pcd = 9
    calibration = Calibration(robot_filename, serial_num, num_pcd)
    calibration.run_calibration(penaly_rotation_y)
