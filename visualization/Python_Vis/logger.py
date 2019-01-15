import sys
import os

proj_dir = os.getcwd()

# log
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        path = os.path.join(proj_dir, 'logs')
        filepath = os.path.join(path, filename)
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def save_logs(path):
    filename = str(os.path.basename(path).split('.')[0]) + '.txt'
    sys.stdout = Logger(filename)