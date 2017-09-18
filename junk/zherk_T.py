## Compute the inverse of a square matrix with complex128 entries.
## Test the performance of two routines: numpy.linalg.inv(),  and scipy.linalg.lapack.ztrtri()
## User specifies the size of the matrix and the number of repetitions to average over. E.g.: python dot_vs_zgeru.py 1024 10

## The test of scipy.linalg.lapack.ztrtri() does not work on BGQ and is commented out. 

import sys
import numpy as np
from scipy.linalg.blas import zherk
from scipy.linalg import triu
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
A = A_real+1.0j*A_imaginary # only transposes; does not conjugate, so use zsymm

C = np.empty((n,n),complex)

alpha = 1.0 + 0.0j
beta  = 1.0 + 0.0j

indices=range(N) # list to loop over.


## Time scipy.blas.zsymm(A_T)
times_zherk_T1 = np.empty(N)
for i in indices:
    start_zherk_T1 = time.time()
    C = zherk(alpha, A.T, beta=1.0, c=-np.identity(n,complex), trans=2, lower=1, overwrite_c=1)
    C = C.T
    end_zherk_T1 = time.time()
    times_zherk_T1[i] = end_zherk_T1 - start_zherk_T1
C_zherk_T1 = C

times_zherk_T2 = np.empty(N)
for i in indices:
    start_zherk_T2 = time.time()
    C = zherk(alpha, A.T, beta=-1.0, c=np.identity(n,complex), trans=2, lower=1, overwrite_c=1)
    C = C.T
    end_zherk_T2 = time.time()
    times_zherk_T2[i] = end_zherk_T2 - start_zherk_T2
C_zherk_T2 = C

times_zherk_T3 = np.empty(N)
for i in indices:
    start_zherk_T3 = time.time()
    C = zherk(alpha, A.T, beta=1.0, c=-np.identity(n,complex).T, trans=2, lower=1, overwrite_c=1)
    C = C.T
    end_zherk_T3 = time.time()
    times_zherk_T3[i] = end_zherk_T3 - start_zherk_T3
C_zherk_T3 = C

times_zherk_T4 = np.empty(N)
for i in indices:
    start_zherk_T4 = time.time()
    C = zherk(alpha, A.T, beta=-1.0, c=np.identity(n,complex).T, trans=2, lower=1, overwrite_c=1)
    C = C.T
    end_zherk_T4 = time.time()
    times_zherk_T4[i] = end_zherk_T4 - start_zherk_T4
C_zherk_T4 = C

print "Equal: "+str(bool(np.allclose(C_zherk_T1, C_zherk_T2) and np.allclose(C_zherk_T2, C_zherk_T3) and np.allclose(C_zherk_T3, C_zherk_T4)))
print "\nzherk_T1:"
print "min  = "+str(times_zherk_T1.min())
print "max  = "+str(times_zherk_T1.max())
print "mean = "+str(times_zherk_T1.mean())
print "\nzherk_T2:"
print "min  = "+str(times_zherk_T2.min())
print "max  = "+str(times_zherk_T2.max())
print "mean = "+str(times_zherk_T2.mean())
print "\nzherk_T3:"
print "min  = "+str(times_zherk_T3.min())
print "max  = "+str(times_zherk_T3.max())
print "mean = "+str(times_zherk_T3.mean())
print "\nzherk_T4:"
print "min  = "+str(times_zherk_T4.min())
print "max  = "+str(times_zherk_T4.max())
print "mean = "+str(times_zherk_T4.mean())
