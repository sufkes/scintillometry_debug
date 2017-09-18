##Give Credits Later

import numpy as np
from mpi4py import MPI
import sys

matrix_size = int(sys.argv[1])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#if rank == 0:
#   data = np.identity(matrix_size)
#else:
#   data = None

for i in np.arange(0,size):
   print i
   if rank == 0:
        data = np.random.rand(matrix_size,matrix_size)
   else:
        data = None

   data = comm.bcast(data, root=0)
