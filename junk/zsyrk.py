## Compute the inverse of a square matrix with complex128 entries.
## Test the performance of two routines: numpy.linalg.inv(),  and scipy.linalg.lapack.ztrtri()
## User specifies the size of the matrix and the number of repetitions to average over. E.g.: python dot_vs_zgeru.py 1024 10

## The test of scipy.linalg.lapack.ztrtri() does not work on BGQ and is commented out. 

import sys
import numpy as np
from scipy.linalg.blas import zsyrk
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
print C_dot

## Time scipy.blas.zsyrk(A)
times_zsyrk = np.empty(N)
#print A.flags['F_CONTIGUOUS']
for i in indices:
    start_zsyrk = time.time()
    C = zsyrk(alpha, A, beta=beta, c=-np.identity(n,complex), trans=0, lower=0)
    end_zsyrk = time.time()
    times_zsyrk[i] = end_zsyrk - start_zsyrk
C_zsyrk = C
print C_zsyrk

### Time scipy.blas.zsyrk(A_fortran)
#times_zsyrk_fortran = np.empty(N)
#A_fortran = np.asfortranarray(A)
##print A_fortran.flags['F_CONTIGUOUS']
#for i in indices:
#    start_zsyrk_fortran = time.time()
#    y_zgemv = zgemv(alpha, a=A, x=np.conj(x), trans=0)
#    end_zsyrk_fortran = time.time()
#    times_zsryk_fortran[i] = end_zsyrk_fortran - start_zsyrk_fortran
   
#invT[:p_eff,:p_eff] = triu(X2[: p_eff, :m].dot(np.conj(X2)[: p_eff, :m].T))
#for jj in range(p_eff):
#    invT[jj,jj] = (invT[jj,jj] - 1.)/2.
#return invT


print "Equal: "+str(bool(np.allclose(C_dot,C_zsyrk)))
print "dot():"
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "zsyrk(A):"
print "min  = "+str(times_zsyrk.min())
print "max  = "+str(times_zsyrk.max())
print "mean = "+str(times_zsyrk .mean())
#print "zsyrk(A_fortran):"
#print "min  = "+str(times_zsyrk_fortran.min())
#print "max  = "+str(times_zsyrk_fortran.max())
#print "mean = "+str(times_zsyrk_fortran.mean())
