# global variables across files used for debugging and profiling

import cProfile

# pr = None

def init():
    global pr
    pr = cProfile.Profile()
