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

## Time numpy.linalg.inv()
times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    C = triu(A.dot(np.conj(A.T)))-np.identity(n,complex)
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot
C_dot = C

## Time scipy.blas.zsymm(A)
times_zherk = np.empty(N)
print A.flags['F_CONTIGUOUS']
for i in indices:
    start_zherk = time.time()
    C = zherk(alpha, A, beta=beta, c=-np.identity(n,complex), trans=0, lower=0, overwrite_c=1)
    end_zherk = time.time()
    times_zherk[i] = end_zherk - start_zherk
C_zherk = C

## Time scipy.blas.zsymm(A_fortran)
times_zherk_fortran = np.empty(N)
A_fortran = np.asfortranarray(A)
print A_fortran.flags['F_CONTIGUOUS']
for i in indices:
    start_zherk_fortran = time.time()
    C = zherk(alpha, A_fortran, beta=beta, c=-np.identity(n,complex), trans=0, lower=0, overwrite_c=1)
    end_zherk_fortran = time.time()
    times_zherk_fortran[i] = end_zherk_fortran - start_zherk_fortran
C_zherk_fortran = C

## Time scipy.blas.zsymm(A_T)
times_zherk_T = np.empty(N)
print A.T.flags['F_CONTIGUOUS']
for i in indices:
    start_zherk_T = time.time()
    C = zherk(alpha, A.T, beta=beta, c=-np.identity(n,complex), trans=2, lower=1, overwrite_c=1)
    C = C.T
    end_zherk_T = time.time()
    times_zherk_T[i] = end_zherk_T - start_zherk_T
C_zherk_T = C


print "Equal: "+str(bool(np.allclose(C_dot,C_zherk) and np.allclose(C_zherk, C_zherk_fortran) and np.allclose(C_zherk_fortran, C_zherk_T)))
print "\ndot():"
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\nzherk(A):"
print "min  = "+str(times_zherk.min())
print "max  = "+str(times_zherk.max())
print "mean = "+str(times_zherk .mean())
print "\nzsyrk(A_fortran):"
print "min  = "+str(times_zherk_fortran.min())
print "max  = "+str(times_zherk_fortran.max())
print "mean = "+str(times_zherk_fortran.mean())
print "\nzsyrk(A_T):"
print "min  = "+str(times_zherk_T.min())
print "max  = "+str(times_zherk_T.max())
print "mean = "+str(times_zherk_T.mean())
