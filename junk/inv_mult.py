import sys
import numpy as np
from numpy.linalg import inv
import time

# Initialize matrices.
n = int(sys.argv[1])
N = int(sys.argv[3])
m = n

# Create random matrices (the same ones each time). 
np.random.seed(42)
A_real = np.random.rand(n,m)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n,m)-0.5
A = A_real+1.0j*A_imaginary
del A_real
del A_imaginary

np.random.seed(44)
B_real = np.random.rand(m,n)-0.5
np.random.seed(45)
B_imaginary = np.random.rand(m,n)-0.5
B = B_real+1.0j*B_imaginary
del B_real
del B_imaginary

# Matrix multiplication.

C = np.empty((n,m),complex)

indices=range(N)
start_mult = time.time()
for i in indices:
    C = np.dot(A,B)
end_mult = time.time()

start_inv = time.time()
for i in indices:
    C = inv(A)
end_inv = time.time()

print "Time per multiplication: "+str((end_mult-start_mult)/N)
print "Time per inversion     : "+str((end_inv-start_inv)/N)
print "Ratio (inv/mult)       : "+str((end_inv-start_inv)/(end_mult-start_mult))

