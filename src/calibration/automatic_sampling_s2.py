import numpy as np
from klampt import vis
from klampt.plan import robotcspace
from klampt.model import ik
from klampt.math import se3, so3, vectorops
from PyRobotBridge.control.generic import Controller
from calibration import *
from tqdm import tqdm
import pandas as pd
import time
import sys


class SampleController(Controller):
    def __init__(self, sample_list, robot, *args, **kwargs):
        super(SampleController, self).__init__(*args, **kwargs)
        self._milestone_count = 0
        self._sample_list = sample_list
        self._robot = robot

    def update(self, state):
        q = state.actual_q
        print("Current configuration is: ", q)
        self._robot.setConfig(self._sample_list[self._milestone_count])
        col_check = self._robot.selfCollides()
        if col_check is True:
            sys.exit("The robot will be in self-collision!")
        error = np.linalg.norm(q - self._sample_list[self._milestone_count][7:13])
        print("We are working on " + str(self._milestone_count) + "th sample milestone.")
        print("The error is: ", error)
        if error <= 8e-5:
            if self._milestone_count == len(self._sample_list) - 1:
                self._milestone_count = len(self._sample_list) - 2
                self._quit = True
            print("Waiting for go to next milestone.")
            for i in tqdm(range(0, 20)):
                time.sleep(1.)
            self._milestone_count += 1
        self.servo(q=self._sample_list[self._milestone_count][7:13])


def solve_ik(robotlink, localpos, worldpos, index):
    """

    :param robotlink: {Klampt.RobotModelLink} -- the Klampt robot link model
    :param localpos: {list} -- the list of points in the robot link frame
    :param worldpos: {list} -- the list of points in the world frame
    :param index: {int} -- the index number
    :return: robot.getConfig(): {list} -- the list of the robot configuration
    """
    robot = robotlink.robot()
    space = robotcspace.RobotCSpace(robot)
    obj = ik.objective(robotlink, local=localpos, world=worldpos)
    maxIters = 100
    tol = 1e-8
    for i in range(1000):
        s = ik.solver(obj, maxIters, tol)
        res = s.solve()
        if res and not space.selfCollision() and not space.envCollision():
            return robot.getConfig()
        else:
            print("Couldn't solve IK problem in " + str(index) +
                  "th. Or the robot exists self-collision or environment collision.")
            s.sampleInitial()


def sample_conf_true(filename):
    """

    :param filename: {string} -- the file name that records the robot configuration
    :return: Clink_set: {list} -- set of configuration
    """
    Clink_set = np.array(pd.read_csv(filename, header=None)).tolist()
    for i in range(len(Clink_set)):
        Clink_set[i] = np.multiply(Clink_set[i], np.pi/180.).tolist()
        Clink_set[i] = [0]*7 + Clink_set[i] + [0]
    return Clink_set


def gen_est_config_list(robot, Clink_set, origin_config):
    """

    :param robot: {Klampt.RobotModel} -- the robot model
    :param Clink_set: {list} -- the strandard robot configuration list
    :param origin_config: {list} -- the actual original start configuration
    :return: est_config_list: {list} -- the estimated configuration list
    """
    est_config_list = []
    est_config_list.append(origin_config)
    robot.setConfig(origin_config)
    origin_trans = robot.link('link_6').getTransform()
    robot.setConfig(Clink_set[0])
    mother_trans = robot.link('link_6').getTransform()
    diff_trans = se3.mul(origin_trans, se3.inv(mother_trans))
    local_pos = [[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1]]
    for i in range(1, len(Clink_set)):
        robot.setConfig(Clink_set[i])
        child_trans = se3.mul(diff_trans, robot.link('link_6').getTransform())
        relative_trans = se3.mul(child_trans, se3.inv(origin_trans))
        robot.setConfig(origin_config)
        world_pos = [se3.apply(relative_trans, robot.link('link_6').getWorldPosition(local_pos[0])),
                     se3.apply(relative_trans, robot.link('link_6').getWorldPosition(local_pos[1])),
                     se3.apply(relative_trans, robot.link('link_6').getWorldPosition(local_pos[2]))]
        q = solve_ik(robot.link('link_6'), local_pos, world_pos, i)
        est_config_list.append(q)
    return est_config_list


def collision_checking(world, est_config_list, density):
    """

    :param world: {Klampt.WorldModel} -- the world model
    :param est_config_list: {list} -- the list of calibration configuration set
    :param density: {float} -- the interpolation step, for example: 0.001
    :return: path: {list or bool} -- if it is list, this is the path of the calibration. If it is bool(False), this
                                     means that the path has collision
    """
    robot = world.robot(0)
    q_0 = robot.getConfig()
    inter_len = int(1./density)
    path = []
    for i in range(len(est_config_list)):
        for j in range(1, inter_len+1):
            if i == 0:
                milestone = vectorops.interpolate(q_0, est_config_list[i], j * density)
            else:
                milestone = vectorops.interpolate(est_config_list[i-1], est_config_list[i], j*density)
            robot.setConfig(milestone)
            if robot.selfCollides() is True:
                return False
            else:
                path.append(milestone)
    return path


def cal_est_config_list(robot, serial_num):
    est_config_list = []
    local_pos = [[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1]]
    Ticp_list = np.load("../../data/suture_experiment/standard_trans_list/trans_array.npy")
    est_calibration = np.load("../../data/suture_experiment/calibration_result_files/" +
                              serial_num + "B/calibration_result.npy")
    Toct = (so3.from_rpy(est_calibration[0:3]), est_calibration[3:6])
    Tneedle = (so3.from_rpy(est_calibration[6:9]), est_calibration[9:12])
    for i in range(0, 9):
        Tee = se3.mul(se3.mul(Toct, se3.inv(Ticp_list[i])), se3.inv(Tneedle))
        world_pos = [se3.apply(Tee, local_pos[0]),
                     se3.apply(Tee, local_pos[1]),
                     se3.apply(Tee, local_pos[2])]
        q = solve_ik(robot.link('link_6'), local_pos, world_pos, i)
        est_config_list.append(q)
    return est_config_list


def sampling_9(serial_num):
    world_name = "flatworld"
    world_file = "../../data/robot_model_files/worlds/" + world_name + ".xml"
    robot_file = "../../data/robot_model_files/robots/irb120_icp.rob"
    world = get_world(world_file, robot_file, visualize_robot=True)
    robot = world.robot(0)
    est_config_list = cal_est_config_list(robot, serial_num)
    path = collision_checking(world, est_config_list, 0.001)
    if path is False:
        sys.exit("The calibration path exist self-collision.")
    else:
        print("The calibration path is collision free.")
        vis.clear()
        vis.add("world", world)
        for i in range(len(est_config_list)):
            vis.add("estimated configure" + str(i), est_config_list[i])
            vis.setColor("estimated configure" + str(i), 1.0, 0.0, 0.1 * i, 0.5)
        vis.spin(float('inf'))
    df = pd.DataFrame(np.asarray(est_config_list)[:, 7:13] * (180. / np.pi))
    df.to_csv("../../data/robot_configuration_files/configuration_record_" + serial_num + ".csv",
              header=False, index=False)
    time.sleep(1.)
    est_config_list = est_config_list[0:]
    controller = SampleController(est_config_list, robot, '192.168.1.178')
    controller.start()


if __name__ == "__main__":
    serial_num = "200228B"
    sampling_9(serial_num)

