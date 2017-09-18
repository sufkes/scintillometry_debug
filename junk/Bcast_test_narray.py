import sys
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = int(sys.argv[1])
m = n
N = int(sys.argv[2])

data = np.zeros((n,m),dtype="float64")

if rank == 0:
    np.random.seed(42)
    data = np.random.rand(n,m)

start = time.time()
for i in range(N):
    comm.Bcast(data, root = 0)
end = time.time()
timePerBcast = (end-start)/N

print "Rank = "+str(rank)+", time per Bcast = "+"{:.3E}".format(timePerBcast)
