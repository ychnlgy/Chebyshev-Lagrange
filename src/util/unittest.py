import sys

from . import color

FUNCTION_NAME = "unittest"

INIT = color.bold("Running %d tests:")
START = color.bold(" >> %s...")
PASS = color.green("OK\n")
FAIL = color.red("X\n")
NO_TESTS = color.red("No tests are written for %s.")
DONE = color.green("All %d tests passed.")

TESTS = []
COUNT = 0

def unittest(Class=None):
    if Class is not None:
        test = Test(Class)
        TESTS.append(test)
        return Class
    else:
        run_tests()

# === PRIVATE ===

def run_tests():
    N = len(TESTS)
    print(INIT % N)
    for test in TESTS:
        test.run()
    print(DONE % N)

class Test:
    
    def __init__(self, Class):
    
        self.name = self.get_classname(Class)
    
        if not hasattr(Class, FUNCTION_NAME):
            print(NO_TESTS % self.name)
            raise SystemExit(1)
    
        self.prog = Class
    
    def run(self):
        sys.stdout.write(START % str(self.name))
        sys.stdout.flush()
        try:
            self.prog.unittest()
            sys.stdout.write(PASS)
        except AssertionError as e:
            sys.stdout.write(FAIL)
            raise
    
    def get_classname(self, Class):
        return str(Class).lstrip("<class '").rstrip("'>")
