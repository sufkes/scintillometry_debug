import sys
from mpi4py import MPI
import numpy as np
#import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = int(sys.argv[1])
m = int(sys.argv[2])

data = np.zeros((n,m),dtype="complex128")

if rank == 0:
    data = data + np.ones((n,m))+1.0j*np.ones((n,m))
    comm.Barrier() # Hopefully synchronize the time at which Bcast is called.
    wstart = MPI.Wtime()
    comm.Send(data, dest=1, tag=0)
    comm.Barrier()
    wend = MPI.Wtime()
elif rank == 1:
    comm.Barrier() # Hopefully synchronize the time at which Bcast is called.
    wstart = MPI.Wtime()
    comm.Recv(data, source=0, tag=0)
    comm.Barrier()
    wend = MPI.Wtime()

wduration = wend - wstart

wall_starts = comm.gather(wstart, root=0)
wall_ends = comm.gather(wend, root=0)
wall_duration = comm.gather(wduration, root=0)

success = bool( np.all(np.real(data)) and np.all(np.imag(data)) and len(data[:,0])==n and len(data[0,:])==m )
all_success = comm.gather(success, root=0)

if rank == 0:
    wfirst_start = min(wall_starts)
    wlast_end = max(wall_ends)
    wmax_duration = max(wall_duration)
    
    total_success = np.all(all_success)

print "Rank = "+str(rank)+", Wtime = "+str(wend-wstart)
comm.Barrier()
if rank == 0:
    print "max individual time = "+str(wmax_duration)
    print "total Wtime = "+str(wlast_end-wfirst_start)
    print "Bcast success = "+str(total_success)
