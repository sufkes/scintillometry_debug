from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def returner(dat):
    return dat

data = np.array([0])
other_data = np.array([2])

if rank == 0:
    data = np.array([1])
    comm.Bcast(data, root = 0)

for i in range(1,size):
    if rank == i:
#        comm.Bcast(returner(data), root=0)
        comm.Bcast(data, root=0)
        comm.Bcast(other_data, root=0)
#        comm.Barrier()
    
print rank, data, other_data

# The entire code will be executed by rank=0. The other ranks will get stuck on the second broadcast, waiting for rank=0 to broadcast the data.
