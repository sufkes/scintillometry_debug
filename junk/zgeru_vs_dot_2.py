## Performs the calculation (matrix A)[start:end, :] = (matrix A)[start:end, :] + (constant)*(row vector)*(column vector) 
## Compare performance using numpy.dot() and scipy.linalg.blas.zgeru()

import sys
import numpy as np
import time
from scipy.linalg.blas import zgeru
import scipy as sp

## Initialize matrices.
n = int(sys.argv[1]) # length of column vector
m = int(sys.argv[2]) # length of row vector
N = int(sys.argv[3]) # number of repetitions of test

## Create arrays. 
np.random.seed(42)
A_real = np.random.rand(n)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n)-0.5
A = A_real+1.0j*A_imaginary # complex "column vector" of length n
del A_real
del A_imaginary

np.random.seed(44)
B_real = np.random.rand(m)-0.5
np.random.seed(45)
B_imaginary = np.random.rand(m)-0.5
B = B_real+1.0j*B_imaginary # complex "row vector" of length m
del B_real
del B_imaginary

np.random.seed(46)
C_real = np.random.rand(n+10,m)-0.5
np.random.seed(47)
C_imaginary = np.random.rand(n+10,m)-0.5
C = C_real+1.0j*C_imaginary # complex matrix with >n rows, m columns
del C_real
del C_imaginary

np.random.seed(46)
D_real = np.random.rand(n+10,m)-0.5
np.random.seed(47)
D_imaginary = np.random.rand(n+10,m)-0.5
D = D_real+1.0j*D_imaginary # complex matrix with >n rows, m columns
del D_real
del D_imaginary

np.random.seed(48)
alpha = (np.random.rand(1) - 0.5) + 1j*(np.random.rand(1) - 0.5) # complex constant

indices=range(N) # list to loop over when repeating test

A_2d = A[:,np.newaxis] # convert vector to 2D arrays
B_2d = B[np.newaxis,:] # convert vector to 2D arrays

start = 4 # row of C, D to start at.
end = 4+n # row of C, D to stop at. 

## Test using numpy.dot()
times_dot = np.empty(N) # array of times for numpy.dot() calculation.
for i in indices:
    start_dot = time.time()
    C[start:end,:] = C[start:end,:] + alpha*np.dot(A_2d,B_2d)
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot

## Test using zgeru()
# The zgeru wrapper apparently requires a column-major order input array, and works fastest when the input array is contiguous in memory. 
# Thus, we transpose the entire equation, and slice in the column dimension (instead of the row dimension); the operation is equivalent. 
# Other, simpler input methods work, but this method is the fastest by far. 
times_zgeru = np.empty(N) # array of times for zgeru() calculation.
for i in indices:
    start_zgeru = time.time()
    zgeru(alpha, B, A, incx=1, incy=1, a=D.T[:,start:end], overwrite_x=0, overwrite_y=0, overwrite_a=1)
    end_zgeru = time.time()
    times_zgeru[i] = end_zgeru - start_zgeru

## Print results
print "results equal: "+str(np.allclose(C,D)) # Check if results are the same.
print "\nnp.dot(): "
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\nzgeru(): "
print "min  = "+str(times_zgeru.min())
print "max  = "+str(times_zgeru.max())
print "mean = "+str(times_zgeru.mean())
