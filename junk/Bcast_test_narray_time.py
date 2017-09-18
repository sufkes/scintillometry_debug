import sys
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = int(sys.argv[1])
m = int(sys.argv[2])

data = np.zeros((n,m),dtype="complex128")

if rank == 0:
    np.random.seed(42)
    data = data + np.ones(n,m)
    
    start = time.time()
    comm.Bcast(data, root = 0)
    end = time.time()
else:
    start = time.time()
    comm.Bcast(data, root = 0)
    end = time.time()

print "Rank = "+str(rank)+", data sample = "+str(data[0,0])+", time = "+str(end-start)
