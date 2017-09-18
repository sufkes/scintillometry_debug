# This script generates a rank mapping text file. It only works for a 64-node block (e.g. a debug block).

import sys
import numpy as np

nodes           = int(sys.argv[1])
rpn             = int(sys.argv[2])
processes       = int(sys.argv[3])
same_node       = bool(int(sys.argv[4]))

rm = np.empty((processes,6), int)

if not same_node:
    rpn = 1

# T column
for process in range(processes):
    rm[process, 5] = process % rpn

# E column
rm[:, 4] = 0
for process in range(rpn, processes, 2*rpn):
    rm[process:min(process+rpn, processes), 4] = 1

# D column
rm[:, 3] = 0
for process in range(2*rpn, processes, 2*2*rpn):
    rm[process:min(process+2*rpn, processes), 3] = 1
    
# C column
rm[:, 2] = 0
for process in range(2*2*rpn, processes, 4*2*2*rpn):
    rm[process:min(process+2*2*rpn, processes), 2] = 1
    if process+2*2*rpn < processes:
        rm[process+2*2*rpn:min(process+2*2*2*rpn, processes), 2] = 2
    if process+2*2*2*rpn < processes:
        rm[process+2*2*2*rpn:min(process+3*2*2*rpn, processes), 2] = 3

# B column
rm[:, 1] = 0
for process in range(4*2*2*rpn, processes, 2*4*2*2*rpn):
    rm[process:min(process+4*2*2*rpn, processes), 1] = 1

# A column 
rm[:, 0] = 0
for process in range(2*4*2*2*rpn, processes, 2*2*4*2*2*rpn):
    rm[process:min(process+2*4*2*2*rpn, processes), 0] = 1

nodes           = int(sys.argv[1])
rpn             = int(sys.argv[2])
processes       = int(sys.argv[3])
same_node       = bool(int(sys.argv[4]))

if same_node:
    same_node_string = "s"
else:
    same_node_string = "d"

filename = "rm_"+same_node_string+"_nodes_"+str(nodes)+"_rpn_"+str(rpn)+"_mpisize_"+str(processes)
np.savetxt(filename, rm, delimiter=" ", fmt="%i")
