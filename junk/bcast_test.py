import os,sys
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = np.double([0])
#    data = {'a':1,'b':2,'c':3}
else:
    data = None

data = comm.bcast(data, root=0)
print rank, data
