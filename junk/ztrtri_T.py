## Compute the inverse of a square matrix with complex128 entries.
## Test the performance of two routines: numpy.linalg.inv(),  and scipy.linalg.lapack.ztrtri()
## User specifies the size of the matrix and the number of repetitions to average over. E.g.: python dot_vs_zgeru.py 1024 10

## The test of scipy.linalg.lapack.ztrtri() does not work on BGQ and is commented out. 

import sys
import numpy as np
from numpy.linalg import inv
from scipy.linalg.lapack import ztrtri
import time
from copy import deepcopy

# Initialize matrices.
n = int(sys.argv[1]) # Size of matrix (nxn)
N = int(sys.argv[2]) # Number of repetitions to average over

# Create random matrix (the same one each time). 
np.random.seed(42)
A_real = np.random.rand(n,n)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n,n)-0.5
A = A_real+1.0j*A_imaginary

A = np.triu(A) # convert to upper triangular form

np.random.seed(42)
B_real = np.random.rand(n,n)-0.5
np.random.seed(43)
B_imaginary = 1.0*np.random.rand(n,n)-0.5
B = B_real+1.0j*B_imaginary

## Test matrix inversion

C = np.empty((n,n),complex) # initialize inverse.

indices=range(N) # list to loop over.

## Time numpy.linalg.inv()
times_inv = np.empty(N)
for i in indices:
    start_inv = time.time()
    C = inv(A)
    end_inv = time.time()
    times_inv[i] = end_inv - start_inv
C_inv = deepcopy(C)

## Time scipy.linalg.lapack.ztrtri()
times_ztrtri = np.empty(N)
print "Input F-contiguous: "+str(A.flags['F_CONTIGUOUS'])
for i in indices:
    start_ztrtri = time.time()
    C = ztrtri(A)[0]
    end_ztrtri = time.time()
    times_ztrtri[i] = end_ztrtri - start_ztrtri
C_ztrtri = deepcopy(C)

## Time scipy.linalg.lapack.ztrtri()
times_ztrtri_T = np.empty(N)
print "Input F-contiguous: "+str(A.T.flags['F_CONTIGUOUS'])
for i in indices:
    start_ztrtri_T = time.time()
    C = ztrtri(A.T,lower=1)[0].T
    end_ztrtri_T = time.time()
    times_ztrtri_T[i] = end_ztrtri_T - start_ztrtri_T
C_ztrtri_T = deepcopy(C)

print "Equal: "+str(bool(np.allclose(C_inv,C_ztrtri) and np.allclose(C_ztrtri,C_ztrtri_T)))
print "inv(A):"
print "min  = "+str(times_inv.min())
print "max  = "+str(times_inv.max())
print "mean = "+str(times_inv.mean())
print "ztrtri(A):"
print "min  = "+str(times_ztrtri.min())
print "max  = "+str(times_ztrtri.max())
print "mean = "+str(times_ztrtri.mean())
print "ztrtri(A.T):"
print "min  = "+str(times_ztrtri_T.min())
print "max  = "+str(times_ztrtri_T.max())
print "mean = "+str(times_ztrtri_T.mean())