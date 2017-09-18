from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.array([0])
other_data = np.array([0])

if rank == 0:
    data = np.array([1])
    comm.send(data, dest=1, tag=0)

for i in range(1,size):
    if rank == i:
        string="Rank "+str(rank)+" about to receive."
        print string
        data = comm.recv(source=0, tag=0)
        string = "Rank "+str(rank)+" passed receive line."
        print string

print rank, data

# Code runs correctly for np=2. When np=3, the rank=0 and rank=1 processes will execute fully; the rank=2 process will get stuck waiting to recieve data from rank=0.
