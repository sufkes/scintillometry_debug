import sys
import numpy as np
import time
from scipy.linalg.blas import zgeru
import scipy as sp

# Initialize matrices.
n = int(sys.argv[1])
N = int(sys.argv[2])


# Create random matrices (the same ones each time). 
np.random.seed(42)
A_real = np.random.rand(n)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n)-0.5
A = A_real+1.0j*A_imaginary
del A_real
del A_imaginary

np.random.seed(44)
B_real = np.random.rand(n)-0.5
np.random.seed(45)
B_imaginary = np.random.rand(n)-0.5
B = B_real+1.0j*B_imaginary
del B_real
del B_imaginary

# Matrix multiplication.

indices=range(N)

A_2d = A[:,np.newaxis]
B_2d = B[np.newaxis,:]

times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    C = np.dot(A_2d,B_2d)
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot

times_outer = np.empty(N)
for i in indices:
    start_outer = time.time()
    D = np.outer(A,B)
    end_outer = time.time()
    times_outer[i] = end_outer - start_outer

times_ein = np.empty(N)
for i in indices:
    start_ein = time.time()
    E = np.einsum('i,j->ij', A, B)
    end_ein = time.time()
    times_ein[i] = end_ein - start_ein

times_spdot = np.empty(N)
for i in indices:
    start_spdot = time.time()
    F = sp.dot(A_2d,B_2d)
    end_spdot = time.time()
    times_spdot[i] = end_spdot - start_spdot

times_spouter = np.empty(N)
for i in indices:
    start_spouter = time.time()
    G = sp.outer(A, B)
    end_spouter = time.time()
    times_spouter[i] = end_spouter - start_spouter

times_zgeru = np.empty(N)
for i in indices:
    start_zgeru = time.time()
    H = zgeru(1, A, B)
    end_zgeru = time.time()
    times_zgeru[i] = end_zgeru - start_zgeru

print "\nnp.dot(): "
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\nnp.outer(): "
print "min  = "+str(times_outer.min())
print "max  = "+str(times_outer.max())
print "mean = "+str(times_outer.mean())
print "\neinsum(): "
print "min  = "+str(times_ein.min())
print "max  = "+str(times_ein.max())
print "mean = "+str(times_ein.mean())
print "\nsp.dot(): "
print "min  = "+str(times_spdot.min())
print "max  = "+str(times_spdot.max())
print "mean = "+str(times_spdot.mean())
print "\nsp.outer(): "
print "min  = "+str(times_spouter.min())
print "max  = "+str(times_spouter.max())
print "mean = "+str(times_spouter.mean())
print "\nzgeru(): "
print "min  = "+str(times_zgeru.min())
print "max  = "+str(times_zgeru.max())
print "mean = "+str(times_zgeru.mean())
