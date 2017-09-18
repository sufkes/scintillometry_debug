import sys
from mpi4py import MPI
import numpy as np
import cProfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = np.array([0])
    comm.Sendrecv(data, dest=1, sendtag=5, recvbuf=data, source=1, recvtag=0)
elif rank == 1:
    data = np.array([1])
    comm.Sendrecv(data, dest=0, sendtag=0, recvbuf=data, source=0, recvtag=5)

print "Rank = "+str(rank)+", data = "+str(data)
