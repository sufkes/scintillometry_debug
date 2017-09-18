## Compute the outer product of two vectors. (nx1 times 1xn -> nxn)
## Test the performance of four routines: numpy.dot(), numpy.outer(), numpy.einsum(), and scipy.linalg.blas.zgeru()
## User specifies the length of the vectors and the number of repetitions to average over. E.g.: python dot_vs_zgeru.py 1024 10

## The test of scipy.linalg.blas.zgeru() does not work on BGQ and is commented out. 

import sys
import numpy as np
import time
#from scipy.linalg.blas import zgeru # Does not work on BGQ


## Initialize matrices.
n = int(sys.argv[1]) # Length of vectors
N = int(sys.argv[2]) # Number of repetitions


## Create vectors with random complex128 entries. 
np.random.seed(42)
A_real = np.random.rand(n)-0.5
np.random.seed(43)
A_imaginary = 1.0*np.random.rand(n)-0.5
A = A_real+1.0j*A_imaginary

np.random.seed(44)
B_real = np.random.rand(n)-0.5
np.random.seed(45)
B_imaginary = np.random.rand(n)-0.5
B = B_real+1.0j*B_imaginary

A_2d = A[:,np.newaxis] # convert vector to 2d array. A_2d is a column vector.
B_2d = B[np.newaxis,:] # convert vector to 2d array. B_2d is a row vector.


## Test outer product.

indices=range(N) # list to loop over

## Time numpy.dot()
start_dot = time.time()
for i in indices:
    C = np.dot(A_2d,B_2d)
end_dot = time.time()

## Time numpy.outer()
start_outer = time.time()
for i in indices:
    D = np.outer(A,B)
end_outer = time.time()

## Time numpy.einsum()
start_ein = time.time()
for i in indices:
    E = np.einsum('i,j->ij', A, B)
end_ein = time.time()

## Time scipy.linalg.blas.zgeru() -- does not work on BGQ
#start_zgeru = time.time()
#for i in indices:
#    F = zgeru(1, A, B)
#end_zgeru = time.time()

print "Time per multiplication np.dot()  : "+str((end_dot-start_dot)/N)
print "Time per multiplication np.outer(): "+str((end_outer-start_outer)/N)
print "Time per multiplication einsum()  : "+str((end_ein-start_ein)/N)
#print "Time per multiplication zgeru()   : "+str((end_zgeru-start_zgeru)/N)
