# toeplitz_decomposition
Applies to code in Steve's GitHub. Written by Aladdin, Visal, Steve. 

### Extracting Data from your binned file ###
To extract binned data, use `extract_realData2.py`, which requires Python 2.7, NumPy, SciPy, and Matplotlib. 

Extract data on CITA (or personal computer for small *n*, *m*), then move the extracted data to SciNet/BGQ.

To extract, use the format:
```
$ python extract_realData2.py binnedDataFile numrows numcols offsetn offsetm n m
```

where *binnedDataFile* is the raw data you wish to extract (must be in the same directory as `extract_realData2.py`). *numrows* and *numcols* refer to the raw data, and must be set correctly for proper extraction. *numrows* is the number of blocks (or the number of frequency bins) and *numcols* is the size of each block (or the number of time bins). The remaining arguments apply to the extracted dynamic spectrum, and can be set as desired. *offsetn* and *offsetm* are the lower bounds of the frequency and time bins, respectively. *n* and *m* are the total number of frequency and time bins desired (or, equivalently, the number of blocks and the size of each block, respectively)

For example, if your data has *numrows* = 16384, *numcols* = 660, and you want *offsetn* = 0, *offsetm* = 0, *n* = 4, *m* = 8, use:
```
python extract_realData2.py weighted_average_0_1_6.bin 16384 660 0 0 4 8
```

This will create the directory `./processedData/gate0_numblock_4_meff_16_offsetn_0_offsetm_0`

Note that if a directory with this name already exists, the data therein will be overwritten without warning when `extract_realData2.py` executes. 

The format of the directory name is: `gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm)`

Note that the value of *m* is doubled in the directory name, but you must use the original value of *m* when you perform the decomposition.

Inside this folder, there will be a `gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm)_toep.npy` file. There will also be *n* npy files. They each represent a block of the Toeplitz matrix. Only the files within this folder are required to perform the decomposition.

There will also be a dat file `gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm).dat`. This serves an unknown purpose, and the decomposition can be performed without it.

The current version of the code generates a number of png files, and a file `data.tar.gz`. These serve unknown purposes, and the decomposition can be performed without them.

### Performing decomposition on BGQ ###

The decomposition can be performed locally, on SciNet, or on BGQ. I only include instructions for the BGQ here. 

Please refer to SciNet [BGQ wiki](https://wiki.scinet.utoronto.ca/wiki/index.php/BGQ) before continuing.

1. Compress the processed data
```
$ cd processedData/
$ tar -zcvf processedData.tar.gz gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm)
```

2. Move the compressed folder into BGQ using your login information. For example, using scp:
```
scp processedData.tar.gz (BGQusername)@bgqdev.scinet.utoronto.ca:~/
```

3. ssh into BGQ
```
$ ssh (BGQusername)@bgqdev.scinet.utoronto.ca -X
```

4. Clone the GitHub repository containing the source code:
```
git clone https://github.com/sufkes/scintillometry.git
```

5. Move the compressed data into the folder containing the source code:
```
mv processedData.tar.gz scintillometry/toeplitz_decomp/processedData/
```

6. Copy the source code and the extracted data to your scratch directory:
```
cp -r scintillometry/toeplitz_decomp/ $SCRATCH
```

7. Move to your scratch directory and extract the data:
```
cd $SCRATCH/toeplitz_decomp/processedData/
tar -zxvf processedData.tar.gz
cd ..
```

8. Write/edit a job script as per the instructions below.

##### Submitting small jobs (using debugjob) #####
9. Copy the template job script `smalljob_template.sh` to a new name:
```
cp smalljob_template.sh smalljob_name.sh
```

10. Edit the copy `smalljob_name.sh` (e.g. with emacs, vi).
* *method* is the decomposition scheme. yty2 is the method used in Nilou's report.
* Set parameters *offsetn*, *offsetm*, *n* and *m* to the values that were used in `extract_realData2.py`. 
* *p* is an integer parameter used in the decomposition (function currently unclear). It can be set to 2*m*, *m*, *m*/2, *m*/4. Fastest results reportedly occur for *p = m*/2 or *p = m*/4. 
* *pad* is a Boolean value which specifies whether or not to use padding (1 or 0; function currently unclear).

* *bg_size* is the number of nodes in the block. This is automatically set to 64 in a debugjob.
* *NP* is the number of MPI processes. **It must be set to 2*n*.**
* *RPN* is the number of MPI processes per node.
* *OMP_NUM_THREADS* is the number of OpenMP threads per MPI process.

The following conditions must hold for the run to execute:
* *NP* = 2*n*
* *NP* ≤ (*RPN* * *bg_size*)
* *RPN* ≤ *NP*
* (*RPN* * *OMP_NUM_THREADS*) ≤ 64 = number of threads per node.

11. Request a debug block:
```
debugjob
```

12. Once inside the debug block, execute the job script:
```
./smalljob_name.sh
```

The run is timed using the `time` function. Execution consists of 2*n* - 1 loops. Results are saved in `./results/`

12. Move results from scratch directory to desired location.

Results are stored in `./results/gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm)_uc.npy`. An empty directory is also created which serves an unkown purpose.

##### Submitting large jobs (using llsubmit) #####

9. Copy the template job script `largejob_template.sh` to a new name:
```
cp largejob_template.sh largejob_name.sh
```

10. Edit the copy `largejob_name.sh`.
* Follow the same instructions as for small jobs.
* For batch jobs, the number of nodes must be specified. *bg_size* = 64, 128, 256, 512, 1024, 2048.
* Set the directory of the source code *sourcedir*.

11. Submit the job:
```
llsubmit ./largejob_name.sh
```

See SciNet [BGQ wiki](https://wiki.scinet.utoronto.ca/wiki/index.php/BGQ) for instructions on monitoring progress.

12. Move results from scratch directory to desired location.

Results are stored in `./results/gate0_numblock_(n)_meff_(mx2)_offsetn_(offsetn)_offsetm_(offsetm)_uc.npy`. An empty directory is also created which serves an unkown purpose.


### Plotting results ###

Visal says the module 'reconstruct' in toeplitz_scint.py can be used to plot results. I have not tested this. 

Visal used the program plot_simulated.py to plot results obtained from the decomposition alongside simulated results. This requires a separate calculation of simulated results. I have not tested this.

Aladdin's GitHub repository contains programs plot_simulated.py and plot_real.py, which are not in my repository. I have not tested these.
