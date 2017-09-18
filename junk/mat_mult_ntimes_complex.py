import sys
import numpy as np
import time

# Initialize matrices.
n = int(sys.argv[1])
m = int(sys.argv[2])
N = int(sys.argv[3])


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

indices=range(N)
start = time.time()
for i in indices:
    C = np.dot(A,B)
#    C = A.dot(B)
end = time.time()

print "Total time: "+str(end-start)
print "Time per multiplication: "+str((end-start)/N)
