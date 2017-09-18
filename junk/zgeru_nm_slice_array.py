import sys
import numpy as np
import time
from scipy.linalg.blas import zgeru
import scipy as sp

# Initialize matrices.
n = int(sys.argv[1])
m = int(sys.argv[2])
N = int(sys.argv[3])

# Create random matrices (the same ones each time). 
np.random.seed(42)
A_real = np.random.rand(n)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n)-0.5
A = A_real+1.0j*A_imaginary
del A_real
del A_imaginary

np.random.seed(44)
B_real = np.random.rand(m)-0.5
np.random.seed(45)
B_imaginary = np.random.rand(m)-0.5
B = B_real+1.0j*B_imaginary
del B_real
del B_imaginary

np.random.seed(46)
C_real = np.random.rand(n,m)-0.5
np.random.seed(47)
C_imaginary = np.random.rand(n,m)-0.5
C = C_real+1.0j*C_imaginary
del C_real
del C_imaginary

np.random.seed(46)
D_real = np.random.rand(n,m)-0.5
np.random.seed(47)
D_imaginary = np.random.rand(n,m)-0.5
D = D_real+1.0j*D_imaginary
del D_real
del D_imaginary

#print np.array_equal(C,D)

np.random.seed(48)
alpha = (np.random.rand(1) - 0.5) + 1j*(np.random.rand(1) - 0.5)

indices=range(N)

A_2d = A[:,np.newaxis]
B_2d = B[np.newaxis,:]


# Tests

times_dot = np.empty(N)
for i in indices:
    start_dot = time.time()
    C = C + alpha*np.dot(A_2d,B_2d)
    end_dot = time.time()
    times_dot[i] = end_dot - start_dot

#print "alpha: "+str(type(alpha))
#print "D: "+str(type(D))+str(D.shape)
#print "A: "+str(type(A))+str(A.shape)
#print "B: "+str(type(B))+str(B.shape)

times_zgeru = np.empty(N)
for i in indices:
    start_zgeru = time.time()
    D[:,:] = zgeru(alpha, A, B, incx=1, incy=1, a=np.array(D[:,:]), overwrite_x=0, overwrite_y=0, overwrite_a=1)
    end_zgeru = time.time()
    times_zgeru[i] = end_zgeru - start_zgeru

print "\nSuccess: "+str(np.allclose(C,D))

print "\nnp.dot(): "
print "min  = "+str(times_dot.min())
print "max  = "+str(times_dot.max())
print "mean = "+str(times_dot.mean())
print "\nzgeru(): "
print "min  = "+str(times_zgeru.min())
print "max  = "+str(times_zgeru.max())
print "mean = "+str(times_zgeru.mean())
