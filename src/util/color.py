HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def _color(cmd, s):
    return cmd + s + ENDC

def bold(s):
    return _color(BOLD, s)

def green(s):
    return bold(_color(OKGREEN, s))

def red(s):
    return bold(_color(FAIL, s))
