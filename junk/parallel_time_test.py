from mpi4py import MPI
import numpy as np
import time 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a = np.zeros((500,500),complex)
b = np.ones((500,500),complex)

if rank == 0:
    def fun_zero(c,d):
        f=c.dot(d)
        return f
    
    c=fun_zero(a,b)
    print c
    
elif rank == 1:
    def fun_one(c,d):
        f=c.dot(d)
        return f
    c=fun_one(a,b)
    print c

elif rank == 2:
    def fun_two(c,d):
        f=c.dot(d)
        return f
    c=fun_two(a,b)
    print c

elif rank == 3:
    def fun_three(c,d):
        f=c.dot(d)
        return f
    c=fun_three(a,b)
    print c
