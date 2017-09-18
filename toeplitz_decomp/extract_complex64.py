import sys
import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
import os	
import mmap
import re
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift
import tarfile

filename = str(sys.argv[1])
num_rows=int(sys.argv[2]) # frequency
num_columns=int(sys.argv[3]) # time
offsetn=int(sys.argv[4]) # offset in freq
offsetm=int(sys.argv[5]) # offset in time
sizen=int(sys.argv[6]) # size of freq = n
sizem=int(sys.argv[7]) # size of freq = m
nump=sizen

if offsetn>num_rows or offsetm>num_columns or offsetn+sizen>num_rows or offsetm+sizem>num_columns:
	print ("Error sizes or offsets don't match")
	sys.exit(1)

a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')

##### change to 1 for padding #####
pad=1
pad2=1
debug=0

neff=sizen+sizen*pad
meff=sizem+sizem*pad

meff_f=meff+pad2*meff

a_input=np.zeros(shape=(neff,meff), dtype='complex64')

a_input[:sizen,:sizem]=np.copy(a[offsetn:offsetn+sizen,offsetm:offsetm+sizem])

##### specifying file directories #####
newdir = "gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s" %(str(sizen),str(meff_f/2),str(offsetn),str(offsetm))
if not os.path.exists("processedData/"+newdir):	
	os.makedirs("processedData/"+newdir)

const=int(pad2*meff/2)

##### ensuring positive definite matrix #####
norm = a.shape[0]*a.shape[1]
a_input=np.sqrt(a_input)

if debug:
	print (a_input,"after sqrt")
a_input[:sizen,:sizem]=np.fft.fft2(a_input,s=(sizen,sizem))

if debug:
	print (a_input,"after first fft")
c = a_input

a_input[0:sizen, meff-int(round(sizem/2.)):meff] =  a_input[0:sizen, int(sizem/2 + 0.5):sizem]
a_input[0:sizen, int(round(sizem/2.)):sizem] = 0+0j

a_input[neff-int(round(sizen/2.)):neff,0:meff] = a_input[int(sizen/2+0.5):sizen, 0:meff]
a_input[int(round(sizen/2.)):sizen, 0:meff] = 0+0j

if debug:
	print (a_input,"after shift")

## inverse Fourier transform 
a_input=np.fft.ifft2(a_input,s=(neff,meff))

if debug:
	print (a_input,"after inverse fft")
a_input=np.power(np.abs(a_input),2)

if debug:
	print (a_input,"after abs^2")
a_input=np.fft.fft2(a_input,s=(neff,meff))
if debug:
	print (a_input,"after third fft")

path="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s" %(str(sizen),str(meff_f/2),str(offsetn),str(offsetm))
mkdir="mkdir "+path
    
epsilon=np.identity(int(meff_f/2))  *1e-3
input_f=np.zeros(shape=(int(meff_f/2), int(sizen*meff_f/2)), dtype=complex)


#################### making blocked toeplitz elements #########################################
if neff == 1:
    neff += 1
for j in np.arange(0,int(neff/2)):
    rows = np.append(a_input[j,:meff-const], np.zeros(pad2*meff*0+const))
    cols = np.append(np.append(a_input[j,0], a_input[j,const+1:][::-1]), np.zeros(pad2*meff*0+const))
    input_f[0:int(meff_f/2),j*int(meff_f/2):(j+1)*int(meff_f/2)] = sp.linalg.toeplitz(cols,rows)
    if j==0:
        input_f[0:int(meff_f/2),j*int(meff_f/2):(j+1)*int(meff_f/2)] = sp.linalg.toeplitz(np.conj(np.append(a_input[j,:meff-const],np.zeros(pad2*meff*0+const))))+epsilon
print ("##########################")
if neff == 1:
    neff -= 1

for rank in np.arange(0,nump):
    size_node_temp=(sizen//nump)*int(meff_f/2)
    size_node=size_node_temp
    if rank==nump-1:
        size_node = (sizen//nump)*int(meff_f/2)+ (sizen%nump)*int(meff_f/2)
    start = rank*size_node_temp
    file_name=path+'/'+str(rank)+".npy"
    np.save(file_name, np.complex64(np.conj(input_f[:,start:start+size_node].T)))


#mm.close()
if debug:
    pad=1
    u=toeplitz_blockschur(input_f[:neff/2*meff_f,:neff/2*meff_f],meff_f,pad)
    print (u[:,(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2])

    
