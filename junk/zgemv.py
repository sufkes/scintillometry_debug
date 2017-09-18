## Compute the inverse of a square matrix with complex128 entries.
## Test the performance of two routines: numpy.linalg.inv(),  and scipy.linalg.lapack.ztrtri()
## User specifies the size of the matrix and the number of repetitions to average over. E.g.: python dot_vs_zgeru.py 1024 10

## The test of scipy.linalg.lapack.ztrtri() does not work on BGQ and is commented out. 

import sys
import numpy as np
from scipy.linalg.blas import zgemv
import time

# Initialize matrices.
n = int(sys.argv[1]) # Rows in matrix
m = int(sys.argv[2]) # Columns in matrix
N = int(sys.argv[3]) # Number of repetitions to average over

# Create random matrix (the same one each time). 
np.random.seed(42)
A_real = np.random.rand(n,m)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n,m)-0.5
A = A_real+1.0j*A_imaginary

np.random.seed(42)
x_real = np.random.rand(n)-0.5
np.random.seed(43)
x_imaginary = 1.0*np.random.rand(n)-0.5
x = x_real+1.0j*x_imaginary

x_2d = x[np.newaxis,:]

## Test matrix inversion

y_dot   = np.empty((n,1),complex) # initialize inverse.
y_zgemv = np.empty((n),complex)
y_zgemv_T = np.empty((n),complex)

alpha = 1.0 + 0.0j

indices=range(N) # list to loop over.

## Time numpy.linalg.inv()
times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    y_dot = A.dot(np.conj(x_2d.T))
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot


## Time scipy.blas.zgemv(A)
times_zgemv = np.empty(N)
#print A.flags['F_CONTIGUOUS']
for i in indices:
    start_zgemv = time.time()
    y_zgemv = zgemv(alpha, a=A, x=np.conj(x), trans=0)
    end_zgemv = time.time()
    times_zgemv[i] = end_zgemv - start_zgemv


## Time scipy.blas.zgemv(A.T)
times_zgemv_fortran = np.empty(N)
A_fortran = np.asfortranarray(A)
#print A_fortran.flags['F_CONTIGUOUS']
for i in indices:
    start_zgemv_fortran = time.time()
    y_zgemv_fortran = zgemv(alpha, a=A_fortran, x=np.conj(x), trans=0)
    end_zgemv_fortran = time.time()
    times_zgemv_fortran[i] = end_zgemv_fortran - start_zgemv_fortran


print "Equal: "+str(bool(np.allclose(y_dot[:,0],y_zgemv) and np.allclose(y_zgemv,y_zgemv_fortran)))
print "dot():"
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "zgemv(A):"
print "min  = "+str(times_zgemv.min())
print "max  = "+str(times_zgemv.max())
print "mean = "+str(times_zgemv .mean())
print "zgemv(A_fortran):"
print "min  = "+str(times_zgemv_fortran.min())
print "max  = "+str(times_zgemv_fortran.max())
print "mean = "+str(times_zgemv_fortran.mean())
