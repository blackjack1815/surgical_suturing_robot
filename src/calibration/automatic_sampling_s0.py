from PyRobotBridge.control.generic import Controller
import numpy as np
import pandas as pd



class RecordController(Controller):
    def __init__(self, serial_num, *args, **kwargs):
        super(RecordController, self).__init__(*args, **kwargs)
        self.serial_num = serial_num

    def update(self, state):
        q = np.array(state.actual_q) * (180./np.pi)
        df = pd.DataFrame([q.tolist()])
        df.to_csv("../../data/robot_configuration_files/configuration_record_" + self.serial_num + 'A.csv',
                  header=False, index=False)
        self.servo(q=q)
        self._quit = True


if __name__ == "__main__":
    serial_num = "200225N"
    controller = RecordController(serial_num, '192.168.1.178')
    controller.start()


