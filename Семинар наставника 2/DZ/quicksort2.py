import numpy as np
from numba import config, njit, threading_layer

config.THREADING_LAYER = 'omp'

#config.NUMBA_DEFAULT_NUM_THREADS = 8

@njit(parallel=True)
def foo(a, b):
    return a + b

x = np.arange(10.)
y = x.copy()
print(x)
print(y)

# this will force the compilation of the function, select a threading layer
# and then execute in parallel
print(foo(x, y))

# demonstrate the threading layer chosen
print("Threading layer chosen: %s" % threading_layer())