import sys
import numpy as np
from scipy.linalg.blas import zdotc, dznrm2
import time
from copy import deepcopy

# Initialize matrices.
n = int(sys.argv[1]) # Rows in matrix
N = int(sys.argv[2]) # Number of repetitions to average over

# Create random matrix (the same one each time). 
np.random.seed(42)
x_real = np.random.rand(n)-0.5
np.random.seed(43)
x_imaginary = 1.0*np.random.rand(n)-0.5
x = x_real+1.0j*x_imaginary

indices=range(N) # list to loop over.

#b.getA2().dot(np.conj(X2.T))

times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    C = x.dot(np.conj(x))
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot
C_dot = deepcopy(C)


#print x.T.flags['F_CONTIGUOUS']
times_zdotc= np.empty(N)
for i in indices:
    start_zdotc = time.time()
    C = zdotc(x.T, x.T)
    end_zdotc = time.time()
    times_zdotc[i] = end_zdotc - start_zdotc
C_zdotc = deepcopy(C)

times_dznrm2 = np.empty(N)
for i in indices:
    start_dznrm2 = time.time()
    C = dznrm2(x.T)**2
    end_dznrm2 = time.time()
    times_dznrm2[i] = end_dznrm2 - start_dznrm2
C_dznrm2 = deepcopy(C)

times_dznrm2_np = np.empty(N)
for i in indices:
    start_dznrm2_np = time.time()
    C = np.square(dznrm2(x.T))
    end_dznrm2_np = time.time()
    times_dznrm2_np[i] = end_dznrm2_np - start_dznrm2_np
C_dznrm2_np = deepcopy(C)

print "Equal: "+str(bool(np.allclose(C_dot,C_zdotc) and np.allclose(C_zdotc,C_dznrm2) and np.allclose(C_dznrm2, C_dznrm2_np)))
print "\ndot():"
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\zdotc(A):"
print "min  = "+str(times_zdotc.min())
print "max  = "+str(times_zdotc.max())
print "mean = "+str(times_zdotc.mean())
print "\dznrm2(A):"
print "min  = "+str(times_dznrm2.min())
print "max  = "+str(times_dznrm2.max())
print "mean = "+str(times_dznrm2.mean())
print "\dznrm2_np(A):"
print "min  = "+str(times_dznrm2_np.min())
print "max  = "+str(times_dznrm2_np.max())
print "mean = "+str(times_dznrm2_np.mean())
