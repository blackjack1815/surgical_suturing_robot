import pandas as pd
import numpy as np
from klampt import *
from klampt.plan import robotcspace
from klampt.model import ik
from klampt.math import so3, se3
from PyRobotBridge.control.generic import Controller

import time


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


def solve_ik(robotlink, localpos, worldpos):
    """

    :param robotlink: {Klampt.RobotModelLink} -- the Klampt robot link model
    :param localpos: {list} -- the list of points in the robot link frame
    :param worldpos: {list} -- the list of points in the world frame
    :return: robot.getConfig(): {list} -- the list of the robot configuration
    """
    robot = robotlink.robot()
    space = robotcspace.RobotCSpace(robot)
    obj = ik.objective(robotlink, local=localpos, world=worldpos)
    maxIters = 100
    tol = 1e-5
    for i in range(1000):
        s = ik.solver(obj, maxIters, tol)
        res = s.solve()
        if res and not space.selfCollision() and not space.envCollision():
            return robot.getConfig()
        else:
            print("Couldn't solve IK problem. Or the robot exists self-collision or environment collision.")
        s.sampleInitial()


class SutureController(Controller):
    def __init__(self, suture_path, *args, **kwargs):
        super(SutureController, self).__init__(*args, **kwargs)
        self._milestone_count = 0
        self._suture_path = suture_path

    def update(self, state):
        q = state.actual_q
        print("Current configuration is: ", q)
        error = np.linalg.norm(q - self._suture_path[self._milestone_count][7:13])
        print("We already finished " +
              str(round(100*self._milestone_count / len(self._suture_path), 2)) + "% suture path.")
        print("The error is: ", error)
        if error <= 7e-3:
            if self._milestone_count == len(self._suture_path) - 1:
                self._milestone_count = len(self._suture_path) - 2
            self._milestone_count += 1
        self.servo(q=self._suture_path[self._milestone_count][7:13])


class PathGenerator:
    def __init__(self, robot_filename, serial_num, rotation, radius):
        """

        :param robot_filename: {string} -- the location of the Klampt's robot files
        :param serial_num: {string} -- the serial number of the calibration
        :param rotation: {int} -- the rotation angle of the needle
        :param radius: {float} -- the radius of the needle
        :return: None
        """
        self.world_file = robot_filename + "/worlds/flatworld.xml"
        self.robot_file = robot_filename + "/robots/irb120_icp.rob"
        self.world = get_world(self.world_file, self.robot_file, visualize_robot=True)
        self.robot = self.world.robot(0)
        self.suture_center = np.array(pd.read_csv("../../data/suture_experiment/suture_center_files/" +
                                                  serial_num + "_suture_center.csv", header=None))
        print(self.suture_center)
        self.calibration_res = np.load("../../data/suture_experiment/calibration_result_files/" +
                                  serial_num + "/calibration_result.npy")
        self.Toct = (so3.from_rpy(self.calibration_res[0:3]), self.calibration_res[3:6])
        self.Tneedle = (so3.from_rpy(self.calibration_res[6:9]), self.calibration_res[9:12])
        self.rotation = rotation
        self.radius = radius
        self.dim_arr = np.array(pd.read_csv("../../data/oct_volume_calibration/" +
                                            serial_num + "/dimension.csv", header=None))
        seed_config = np.multiply([-92.1441, 60.1883, 22.6847, 1.83303, -82.2083, 59.7906], np.pi / 180.).tolist()
        self.robot.setConfig([0.] * 7 + seed_config + [0.])
        self.robot.link('tool0').setParentTransform(self.Tneedle[0], self.Tneedle[1])

    def set_path(self):
        """

        :return: config_list: {list} -- the list of suturing path in robot configuration space
        """
        config_list = []
        localpos = [[0.0, 0.0, 0.0],
                    [-self.radius + self.radius * np.cos((120 * np.pi) / 180.),
                     -self.radius * np.sin((120 * np.pi) / 180.), 0.0],
                    [-self.radius + self.radius * np.cos((240 * np.pi) / 180.),
                     -self.radius * np.sin((240 * np.pi) / 180.), 0.0]]
        center = [(self.dim_arr[0][0] * 1e-3) -
                  ((self.suture_center[0, 0] / self.dim_arr[1][0]) * self.dim_arr[0][0] * 1e-3),
                  (self.suture_center[1, 0] / self.dim_arr[1][1]) * self.dim_arr[0][1] * 1e-3]
        print("The center is: ", (self.suture_center[2, 0]/self.dim_arr[1][2]) * 1e-3)
        for i in range(0, self.rotation * 10):
            current_angle = i * 0.1
            pos_x_0 = center[0] - self.radius * np.cos((current_angle * np.pi) / 180.)
            pos_y_0 = center[1] + self.radius * np.sin((current_angle * np.pi) / 180.)
            pos_x_1 = center[0] - self.radius * np.cos(((current_angle - 120) * np.pi) / 180.)
            pos_y_1 = center[1] + self.radius * np.sin(((current_angle - 120) * np.pi) / 180.)
            pos_x_2 = center[0] - self.radius * np.cos(((current_angle - 240) * np.pi) / 180.)
            pos_y_2 = center[1] + self.radius * np.sin(((current_angle - 240) * np.pi) / 180.)
            world_pos = [se3.apply(self.Toct, [pos_x_0, pos_y_0, (self.suture_center[2, 0]/self.dim_arr[1][2]) * 1e-2]),
                         se3.apply(self.Toct, [pos_x_1, pos_y_1, (self.suture_center[2, 0]/self.dim_arr[1][2]) * 1e-2]),
                         se3.apply(self.Toct, [pos_x_2, pos_y_2, (self.suture_center[2, 0]/self.dim_arr[1][2]) * 1e-2])]
            q = solve_ik(self.robot.link('tool0'), localpos, world_pos)
            config_list.append(q)
        return config_list


if __name__ == '__main__':
    robot_filename = "../../data/robot_model_files"
    serial_num = "200229L"
    rotation = 150
    radius = 3.961 * 1e-3
    path = PathGenerator(robot_filename, serial_num, rotation, radius)
    config_list = path.set_path()
    # config_list = config_list[::-1]
    print("Suturing path generated.")
    time.sleep(5.)
    controller = SutureController(config_list, '192.168.1.178')
    controller.start()
