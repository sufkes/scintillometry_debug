import os, sys
import numpy as np

A_dir = str(sys.argv[1])
B_dir = str(sys.argv[2])

A = np.load(A_dir)
B = np.load(B_dir)

print "array_equal() : "+str(np.array_equal(A,B))
print "allclose()    : "+str(np.allclose(A, B, atol=1e-15))
print "max(real(A-B)): "+str(np.max(np.real(A-B)))
print "max(real(B-A)): "+str(np.max(np.real(B-A)))
print "max(imag(A-B)): "+str(np.max(np.imag(A-B)))
print "max(imag(B-A)): "+str(np.max(np.imag(B-A)))
