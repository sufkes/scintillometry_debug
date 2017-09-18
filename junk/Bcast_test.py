from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def returner(dat):
    return dat

if rank == 0:
    data = np.array([1])
else:
    data = np.array([0])
    
comm.Bcast(returner(data), root=0)

print rank, data
