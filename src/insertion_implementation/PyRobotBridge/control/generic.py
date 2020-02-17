import socket
from struct import pack, unpack, unpack_from, calcsize
import logging

import numpy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# header field format
HEADER_FIELDS = [
    ('sequence',    '<i'),
]

# payload matrix format
PAYLOAD_MATRICES = [
    ('target_q', (6,), numpy.float64),
    ('actual_q', (6,), numpy.float64),
    ('target_qd', (6,), numpy.float64),
    ('actual_qd', (6,), numpy.float64),
    ('target_x', (6,), numpy.float64),
    ('actual_x', (6,), numpy.float64),
    ('target_xd', (6,), numpy.float64),
    ('actual_xd', (6,), numpy.float64),
]

# ref: http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def _mul(l):
    p = 1
    for x in l:
        p *= x
    return p

def _read_array(f, shape, dtype):
    return numpy.fromfile(f, dtype=dtype, count=_mul(shape)).reshape(shape)

class Controller(object):
    def __init__(self, host, **kwargs):
        self._quit = True

        self._robot_host = host
        self._robot_port = kwargs.pop('port', 30004)

        self._average_interval = kwargs.pop('_average_interval', 0.004)

        self._filters = kwargs.pop('filters', [])

        self._state = None
        self._version = None

    def start(self):
        # open connection to robot
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((self._robot_host, self._robot_port))

        self._quit = False
        while not self._quit:
            # read the encoded state
            n = unpack('<Q', sock.recv(8))[0]
            data = sock.recv(n)

            # decode the state
            state = {}
            offset = 0

            for (k, fmt) in HEADER_FIELDS:
                state[k] = unpack_from(fmt, data, offset)[0]
                offset += calcsize(fmt)

            for (k, shape, dtype) in PAYLOAD_MATRICES:
                n = _mul(shape)
                state[k] = numpy.frombuffer(data, dtype, n, offset)
                offset += n * dtype(1).itemsize

            if 'timestamp' not in state:
                # fake timestamp
                state['timestamp'] = state['sequence'] * self._average_interval

            state = Bunch(**state)

            # invoke update
            self.update(state)

            # run installed filters
            for f in self._filters:
                f(state)

            # check for dropped setpoint
            if self._servo_setpoint is None:
                logger.warn('missing setpoint -> stopping')
                break

            # encode setpoint
            (is_joint, pos, vel) = self._servo_setpoint
            self._servo_setpoint = None

            if is_joint:
                data = b'\x01'
                encoding = [(6, pos), (6, vel)]
            else:
                data = b'\x00'
                encoding = [(16, pos), (6, vel)]

            for (n, x) in encoding:
                if x is None:
                    data += b'\x00'
                else:
                    arr = numpy.array(x).astype(numpy.float64).reshape((-1,))
                    if len(arr) != n:
                        raise RuntimeError('invalid position/velocity dimension')
                    data += b'\x01' + arr.tostring()

            data = pack('<Q', len(data)) + data
            sock.sendall(data)

        self._quit = True

        sock.close()

    def servo(self, q=None, qd=None, x=None, xd=None):
        if q is not None or qd is not None:
            if x is not None or xd is not None:
                raise RuntimeError('only pure joint or pure Cartesian commands allowed')

            self._servo_setpoint = (True, q, qd)
        elif x is not None or xd is not None:
            if q is not None or qd is not None:
                raise RuntimeError('only pure joint or pure Cartesian commands allowed')

            self._servo_setpoint = (False, x, xd)
        else:
            raise RuntimeError('missing q, qd, x, or xd')

    def stop(self):
        self._quit = True

    def update(self, state):
        pass

    @property
    def version(self):
        return self._version

class WiggleController(Controller):
    def __init__(self, *args, **kwargs):
        self._frequency = kwargs.pop('frequency', 0.1)
        self._amplitude = kwargs.pop('amplitude', 5)

        super(WiggleController, self).__init__(*args, **kwargs)

        self._start_time = None
        self._last_t = 0

    def update(self, state):
        wait_time = 1
        from math import pi, cos

        if self._start_time is None:
            self._start_time = state.timestamp

        t = state.timestamp - self._start_time

        if t < wait_time:
            new_target_speed = [0]*6
        else:
            f = self._frequency
            A = self._amplitude/180.0*pi

            new_target_speed = [(-2*pi*A*f)*cos(2*pi*f*(t - wait_time))]*6

        self.servo(qd=new_target_speed)

        dt = t - self._last_t
        self._last_t = t
        print('{:6.4f} {:6.4f}'.format(t, dt), ' '.join(['{:6.2f}'.format(x/pi*180) for x in state.actual_q]))

class ConfigController(Controller):
    def __init__(self, *args, **kwargs):
        self._target = kwargs.pop('target')

        super(ConfigController, self).__init__(*args, **kwargs)

        self._start_time = None
        self._last_t = 0

    def update(self, state):
        from math import pi

        if self._start_time is None:
            self._start_time = state.timestamp

        t = state.timestamp - self._start_time

        self.servo(q=self._target)

        dt = t - self._last_t
        self._last_t = t
        print('{:6.4f} {:6.4f}'.format(t, dt), ' '.join(['{:6.2f}'.format(x/pi*180) for x in state.actual_q]))

if __name__ == '__main__':
    logging.basicConfig()

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='PyUniversalRobot wiggle test', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('robot', help='robot IP address')
    parser.add_argument('--frequency', '-f', type=float, help='wiggle frequency (Hz)', default=0.1)
    parser.add_argument('--amplitude', '-a', type=float, help='wiggle amplitude (deg)', default=5)

    args = parser.parse_args()

    ctrl = WiggleController(args.robot, frequency=args.frequency, amplitude=args.amplitude)
    ctrl.start()
