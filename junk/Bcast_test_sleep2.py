from mpi4py import MPI
import numpy as np
import time
import cProfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.array([0])

def awake_send(a):
    start = time.time()
    comm.Bcast(a, root = 0)
    end = time.time()
    duration = end - start
    return duration

def sleep_receive(a):
    time.sleep(3)
    start = time.time()
    comm.Bcast(a, root = 0)
    end = time.time()
    duration = end-start
    return duration

if rank == 0:
    data = np.array([1])
    profileName = "time_sleep_rank"+str(rank)
    cProfile.run('awake_send(data)',profileName)
else:
    profileName = "time_sleep_rank"+str(rank)
    cProfile.run('sleep_receive(data)',profileName)

duration = "GLUESTICK"
print "Rank = "+str(rank)+", data = "+str(data)+", time = "+str(duration)


# This script demonstrates that (at least in some cases), Bcast is nonblocking for the sender.
