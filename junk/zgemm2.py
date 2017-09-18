import sys
import numpy as np
from scipy.linalg.blas import zgemm
import time
from copy import deepcopy

# Initialize matrices.
n = int(sys.argv[1]) # Rows in matrix
k = int(sys.argv[2])
m = int(sys.argv[3]) # Columns in matrix
N = int(sys.argv[4]) # Number of repetitions to average over

# Create random matrix (the same one each time). 
np.random.seed(42)
C_real = np.random.rand(n,m)-0.5
np.random.seed(43)
C_imaginary = 1.0*np.random.rand(n,m)-0.5
C = C_real+1.0j*C_imaginary

np.random.seed(44)
A_real = np.random.rand(n+10,k)-0.5
np.random.seed(45)
A_imaginary = 1.0*np.random.rand(n+10,k)-0.5
A = A_real+1.0j*A_imaginary

np.random.seed(46)
B_real = np.random.rand(k,m)-0.5
np.random.seed(47)
B_imaginary = 1.0*np.random.rand(k,m)-0.5
B = B_real+1.0j*B_imaginary

alpha = 1.0
beta  = 1.0

start = 4
end = 4 + n

indices=range(N) # list to loop over.

#B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)

times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    C = A[start:end,:].dot(np.conj(B.T))
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot
C_dot = deepcopy(C)
C = C_real+1.0j*C_imaginary


times_zgemm = np.empty(N)
for i in indices:
    start_zgemm = time.time()
    C = zgemm(alpha=1.0, a=A[start:end,:], b=B, trans_b=2)
    end_zgemm = time.time()
    times_zgemm[i] = end_zgemm - start_zgemm
C_zgemm = deepcopy(C)
C = C_real+1.0j*C_imaginary


times_zgemm_T = np.empty(N)
print B.T.flags['F_CONTIGUOUS']
print A.T[:,start:end].flags['F_CONTIGUOUS']
for i in indices:
    start_zgemm_T = time.time()
    C = zgemm(alpha=1.0, a=B.T, b=A.T[:,start:end], trans_a=2).T
    end_zgemm_T = time.time()
    times_zgemm_T[i] = end_zgemm_T - start_zgemm_T
C_zgemm_T = deepcopy(C)
C = C_real+1.0j*C_imaginary


print "Equal: "+str(bool(np.allclose(C_dot,C_zgemm) and np.allclose(C_zgemm,C_zgemm_T)))
print "\ndot():"
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\nzgemm(A):"
print "min  = "+str(times_zgemm.min())
print "max  = "+str(times_zgemm.max())
print "mean = "+str(times_zgemm.mean())
print "\nzgemm(A.T):"
print "min  = "+str(times_zgemm_T.min())
print "max  = "+str(times_zgemm_T.max())
print "mean = "+str(times_zgemm_T.mean())
