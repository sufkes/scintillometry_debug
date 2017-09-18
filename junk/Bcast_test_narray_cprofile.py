import sys
from mpi4py import MPI
import numpy as np
import cProfile
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = int(sys.argv[1])
m = int(sys.argv[2])

data = np.zeros((n,m),dtype="float64")

if rank == 0:
    np.random.seed(42)
    data = np.random.rand(n,m)
    profileName = "time_Bcast_array_rank"+str(rank)
    start = time.time()
    cProfile.run('comm.Bcast(data, root = 0)',profileName)
    end = time.time()
elif (1 <= rank <= 4):
    profileName = "time_Bcast_array_rank"+str(rank)
    start = time.time()
    cProfile.run('comm.Bcast(data, root = 0)',profileName)
    end = time.time()
else:
    comm.Bcast(data, root = 0)

print "Rank = "+str(rank)+", data sample = "+str(data[0,0])+", time = "+str(end-start)
