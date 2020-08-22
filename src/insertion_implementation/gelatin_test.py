import numpy as np
from klampt import *
from klampt.plan import robotcspace
from klampt.model import ik
from PyRobotBridge.control.generic import Controller
from klampt.math import so3, se3
from transformations import rotation_matrix, quaternion_from_matrix, quaternion_inverse, quaternion_multiply, quaternion_slerp, quaternion_matrix


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


class DALKController(Controller):
    def __init__(self, suture_path, xyzrpy, *args, **kwargs):
        super(DALKController, self).__init__(*args, **kwargs)
        self._milestone_count = 0
        self._suture_path = suture_path
        self._xyzrpy = xyzrpy

    def update(self, state):
        x = state.actual_x
        print("Current config ", x)
        error = np.linalg.norm(x - xyzrpy[self._milestone_count])
        print("We already finished " +
              str(round(100*self._milestone_count / len(self._suture_path), 2)) + "% suture path.")
        print("The error is: ", error)
        if error <= 7e-3:
            if self._milestone_count == len(self._suture_path) - 1:
                self._milestone_count = len(self._suture_path) - 2
            self._milestone_count += 1
        self.servo(x=self._suture_path[self._milestone_count])


def dalk_path_generation(path_filename, calib_filename):
    data = np.loadtxt(path_filename)
    oct2robot = np.loadtxt(calib_filename)[0, :].reshape((4, 4))
    robot2oct = np.linalg.inv(oct2robot)
    robot_target = []
    xyzrpy = []
    # dalk_frame = (so3.from_rpy([0, 0, -np.pi/2]), [52.5 * 0.0254, 6.5 * 0.0251, 0.75 * 0.0254])
    dalk_frame = (so3.from_rpy([0, np.pi, 0]), [0.0, 0.0, 0.0])
    for row in data[::1]:
        x_target = row[12 + 6:12 + 6 + 6]
        # xyzrpy.append(x_target)
        T = np.eye(4)
        T = rotation_matrix(x_target[3], [1, 0, 0]).dot(rotation_matrix(x_target[4], [0, 1, 0])).dot(
            rotation_matrix(x_target[5], [0, 0, 1]))
        T[:3, 3] = x_target[0:3]
        T = robot2oct.dot(T)
        T[:3, 3] = T[:3, 3] * 1
        T = oct2robot.dot(T)
        robot_target.append(T)
        T2rpyxyz = list(se3.from_homogeneous(T)[1]) + list(so3.rpy(se3.from_homogeneous(T)[0]))
        print(T2rpyxyz)
        xyzrpy.append(T2rpyxyz)

        # T_m = se3.mul(se3.inv(dalk_frame), se3.from_homogeneous(T))
        # T_h = se3.homogeneous(T_m)
        # T2rpyxyz = list(T_m[1]) + list(so3.rpy(T_m[0]))
        # xyzrpy.append(T2rpyxyz)
        # robot_target.append(T_h)
    return robot_target, xyzrpy


if __name__ == '__main__':
    path_filename = '1_path_6661944144048700.txt'
    calib_filename = '1_calib_6661944144048700.txt'
    pose_list, xyzrpy = dalk_path_generation(path_filename, calib_filename)
    print("Suturing path generated.")
    time.sleep(5.)
    controller = DALKController(pose_list, xyzrpy, '192.168.1.178')
    controller.start()
