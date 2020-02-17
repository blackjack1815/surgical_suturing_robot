import sys
import socket
from struct import pack, unpack, unpack_from, calcsize
import logging

from collections import OrderedDict

import numpy
import yaml

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

class Observer(object):
    def __init__(self, host, **kwargs):
        self._quit = True

        self._robot_host = host
        self._robot_port = kwargs.pop('port', 30004)

        self._average_interval = kwargs.pop('_average_interval', 0.004)

        self._fields = {
            'timestamp': 'DOUBLE',
            'target_q': 'VECTOR6D',
            'actual_q': 'VECTOR6D',
            'target_qd': 'VECTOR6D',
            'actual_qd': 'VECTOR6D',
            'target_x': 'VECTOR6D',
            'actual_x': 'VECTOR6D',
            'target_xd': 'VECTOR6D',
            'actual_xd': 'VECTOR6D',
        }

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

        self._quit = True

        sock.close()

    def stop(self):
        self._quit = True

    def update(self, state):
        pass

    @property
    def state(self):
        return self._state

    @property
    def fields(self):
        return self._fields

    @property
    def version(self):
        return self._version

class LoggingObserver(Observer):
    def __init__(self, *args, **kwargs):
        self._count = kwargs.pop('count', None)

        super(LoggingObserver, self).__init__(*args, **kwargs)

    def update(self, state):
        if self._count is not None and self._count <= 0:
            self.stop()
            return

        record = OrderedDict()
        # build default and joint fields
        for f in self.fields.keys():
            record[f] = getattr(state, f)
            if hasattr(record[f], 'tolist'):
                record[f] = record[f].tolist()

        yaml.dump_all([record], sys.stdout, explicit_start=True)

        if self._count is not None:
            self._count -= 1
            if self._count <= 0:
                self.stop()

if __name__ == '__main__':
    logging.basicConfig()

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='PyRobotBridge observer utility', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('robot', help='robot IP address')
    parser.add_argument('--count', '-c', type=int, help='limit number of records to log')

    args = parser.parse_args()

    # to faciliate logging
    # https://stackoverflow.com/a/8661021
    yaml.add_representer(OrderedDict, lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items()))

    obs = LoggingObserver(args.robot, count=args.count)
    obs.start()
