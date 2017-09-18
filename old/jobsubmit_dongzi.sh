#!/bin/sh
RPN=2
offn=3584
offm=0
n=64
m=1024
p=$((${m}/2))
#p=$(($m))
code_dir=/scratch/p/pen/fleaf5/Alladin_Scintillometry/src
# @ job_name           = test_m_newtoeplitz
# @ job_type           = bluegene
# @ comment            = "n=250, m=150, zero-padded"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 1:00:00
# @ bg_connectivity    = Torus
# @ queue 
nodes=64
NP=$((${nodes}*${RPN}))
OMP=$((16/${RPN})) ## Each core has 4 threads. Since RPN = 16, OMP = 4?
module purge
module unload mpich2/xl
source /scratch/s/scinet/nolta/venv-numpy/bin/activate
module load python/2.7.3 xlf/14.1 essl/5.1 bgqgcc/4.8.1
module load binutils/2.23 mpich2/gcc-4.8.1 
export OMP_NUM_THREADS=${OMP}

cd ${code_dir} 
echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "n ${n}, m ${m}, p ${p},  bg ${nodes}, np ${NP}, rpn ${RPN}, omp ${OMP}"
time runjob --np ${NP} --ranks-per-node=${RPN} --envs HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.4-gcc4.8.1/lib:/scinet/bgq/tools/binutils-2.23/lib64:/scinet/bgq/tools/Python/python2.7.3-20131205//lib:/opt/ibmcmp/lib64/bg/bglib64/:/bgsys/drivers/ppcfloor/gnu-linux/powerpc64-bgq-linux/lib/:/opt/ibmcmp/xlf/bg/14.1/bglib64:/opt/ibmmath/essl/5.1/lib64:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/ : `which python` run_real.py yty2 ${offn} ${offm} ${n} ${m} ${p} 1
#PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/: \

echo "ENDED"

