
# Linux Intro
## Getting Help
Add line argument `-h` or `--help`
## Orienting on a system
### Listing directory contents
`ls` # List all files
#### Arguments
`-a` # Include add all hidden files	`-t` # Sort by date
`-l` # Obtain detailed info on all files	`-h` # File size in human format
**Combining all together**
`ls -alth` 
### Navigating System
`cd <DIR_PATH>` #Connect to the follwing 
`cd..` # Move up one directory `cd` # Go back to home directory
### Creating and Removing Directories
`mkdir <DIR_NAME>` # Create a directory 
`rmdir <DIR_NAME>` # Remove a directory
### Deleting Files
`rm <FILE_NAME>` # Delete File
`rm -r <DIR_NAME>` # Delete recursviely (delete everything inside the dir)
`rm -f <DIR_NAME>` # Delete without **confirmation**
### Copying and renaming files or directories
`cp <source_path> <destination_dir>`
#### Arguments
- `-i` # Interactive mode (Prompts before overwritting)
- `-v` # Add verbose (Shows progress)
- `-r` #Copy recursively

`mv <source_path> <dest_path>` #Move file/dir to a desintation path
`mv <oldname> <newname>` Change name of file/dir
## File permmissions
**Type**
-   `-`: Regular file
-   `d`: Directory
-   `l`: Symbolic link
-   `c`: Character device
-   `b`: Block device

**Permission**
`r`: read 	`w`: write 	`x`: excute
### How to interpert
- First letter:  File Type
- 2nd ~ 4th: **Owner**
- 5th ~ 7th:  **Group**
- 8th ~ End: **Everyone**
## Viewing and editing files
### Viewing
`less <file>`
### Comparing different files
`diff <file1> <file2>`
### Search within a file
`grep <EXPRESSION> <FILE>`
#### Example of expression 
- `*`: Any charcter  (file*)
- [num1-num2]: Range of numbers (ex [1-2][0-9]: 10~29) 
# Storage and file management
## Storage Types 
- **Home**: Source Code / small paramter files / job submission scripts
- **Project**: Large data that should be shared / Has to be static
- **Scratch**:  Intensive read/write on large files / Creat temporary outputs
- **Slurm_TMPDIR**: Temporary usage during the allocated job / Increased performance
## Project Space Consumption
`diskusage_report` #Shows total quota of the group
`lfs quota -u $USER /project` # Shows space / file count use for one user on the entire project
`lfs find <path_to_dir> -type f | wc -l` #Find the number of files in a given directory

## Transferring Data
- Globus: https://www.globus.org/

# Archiving and Compressing files
## Compressing and decompressing
`tar --create --(xz/gzip) --file <NAME_OF_ARCHIVE FILE> <DIR_NAME>.tar.(xz/gzip)` # Archive a directory
`tar --extract --(xz/gzip) --file <NAME_OF_ARCHIVE FILE>.tar.(xz/gzip)` # Extract an archived file

**xz**: Better compression ratio but requires more RAM while working
**gzip**: Less efficient compression but less computation heavy
### Common Options
- `-c` or `--create`: Create a new archive
- `-f` or `--file`: Folliwng is the archive file name
- `-x` or `--extract`: Extract files from archive
- `t` or `--list`: List the content of an archive file
- `-J` or `--xz`: Use `xz`
- `-z` or `--zip`: Use `gzip`

** Usefule shortcut**
 `tar -c(J/z)f <COMPRESSED_FILE_NAME> <SOURCE_FILE/DIR>` # Compression
 `tar -x(J/z)f <FILE_NAME>` # Extraction
## Usage
### Adding multiple directories
`tar -cvf <FILE_NAME> <LIST OF DIR>` # Archive listed directories
`tar -cvf <File_NAME> r*` # Archive all folders that starts with an r
### Appending files
`tar -rf <TAR_FILE_NAME> <FILE_TO_ADD>` # Append a file to an existing archive (**Can't Add files to Compressed Tar files**)
### Check Information
`tar -tvf <TAR_FILE_NAME>`
### Combine two tar files
`tar -A -f <DESINTATION_TAR_FILE> <SOURCE_TAR_FILE>` # Combine the source file to the destination tar file
### Excluding certain files
`tar -cvf <TAR_FILE_NAME> --exclude=<FILE_INFO>` #Exclude certain files from the tar file

## Compressing files and archives
`gzip/bzip2 <NAME_OF_FILE>`

## Unpacking Compressed files and archives
`tar -xvf <TAR_FILE> -C <DEST_DIR>` # Decompress the tar file into the given destintation
**Arguments**
- `-x`:  Extract
- `-v`:  Verbose
- `-C`: Change directory (make sure the directory exists)

### Decompressing gz and bz2 files
`gunzip/bunzip2 <FILE>`

### Extracting a compressed archive file into another directory
 **Arguments**
 `-z`: gz files `j`: bz2 files
`tar -xv(z/j)f <TAR_FILE> -C <DEST>`

### Extracting one file from an archive or a compress archive
`tar -C <DEST_DIR_PATH> -x(j/z)f =<NAME_OF_TAR_FILE> <PATH_OF_FILE_TO_EXTRACT>` # Extracts the specific file to the new directory (add arguments if the tar file is compressed)
### Extracting multiple files
`tar -C <DEST_DIR_PATH> -xv(-j/z)f =<NAME_OF_TAR_FILE> --wildcards "<PATH_OF_FILES>"` # Extract the chosen files into the destination

## Contents of archive files
### Listing the contents
`tar -t(v)f <FILE>` # Provides the content without unpacking (with meta data if v added)
`tar -tf <FILE> |wc -l` # Calculate the number of files inside the tar file
**l**: Passes the output of the command to the next command

### Searching content
`tar -tf <FILE> |grep -a <FILE_TO_SEARCH>` # Searches the file inside the tar file without unpacking

## Other utilties
`du -sh <FILE>` # Checks the size of file/dir/archive
`split -d -b <SIZE-in-MB> <FILE_NAME> <PREFIX_NAME>`  # Split tar files into smaller chunkcs (-d for numer suffices)
`cat <PREFIX> * > <FILE_NAME>.tar` # Recover original file

# Handling Large Collection of Files

## Finding folders with lots of files
`for FOLDER in $(find . -maxdepth 1 -type d | tail -n +2); do
  echo -ne "$FOLDER:\t"
  find $FOLDER -type f | wc -l
done`
## Finding folders using the most disk space
` du -sh  * | sort -hr | head -10`

# Modules
## Module Commands
`module command [options]` # Normal Syntanx
### Sub-command `spider`
`module spider` #Searches the complete tree of all modules and display it
`module spider [APPLICATION]` # Show the list of available versions
`module spider [APP]/[Version]` # Show the list of required modules to access the version
### Sub-command `avail`
`module avail` # Lists the modules that can be loaded
`module avail [LIB]` # List of modules avialble for a particulary library
### Sub-command  `list`
`module list` # Lists the currently loaded modules in the environment
### Sub-command `load`
`module load [module]/[ver]` # Loads the given module to the env
### Sub-command `unload`
`module unload [module]/[ver]` # Removes a module from the env
### Sub-command `purge`
`module purge` # Remove ALL modules in the env
### Sub-command `show/help/whatis`
`module (show/help/whatis) [module]/[lib]` # Gets info on the module
- Show: displays entire model
- help: display help message
- whatis: shows a description
### Sub-command `apropos` or `keyword`
`module apropos/keyword [MODULE]` # Search for a keyword in all modules
## Loading modules automatically
It is not advised to load modules automitcally in .bashrc. It is recommended to load modules only required in job scripts using module collection
## Module Collections
Load the module and save the collection as:
`module save <COLLECTION_NAME>`
`module restore <COLLECTION_NAME>` # Restore the collection 
## Module hierarchy
Load the parent module first and then load the childrens
**Example**:
`module load gcc/9.3 openmpi/4.0 fftw/3.8 `

# Running Jobs
For the clusers Slurm Workload Manager schedulers are used.
## Submitting Jobs
`sbatch <script>.sh` # Submit a job
**Minimal Job script** 
```
#!/bin/bash
#SBATCH --time=<TIME> # resevering time spots
#SBATCH --account=def-<SOME_USER>
echo "Hello, world!"
sleep 30
```
`sbatch --time=<Time_limit> <script>.sh` # Adding a time limit running a script
`--mem-per-cpu=<MEMMORY>` # Allocating memory to the cpu
`--mem<MEMORY>` # Allocating memory to the node

## List jobs
`squeue` # supplies information about all jobs
`sq` # list only "your own" job
** Don't** run sq from a script or program at high frequnecy
## Output
output is placed in a file named `slurm-` with the job ID number and `.out`. 
Use `--output` to specify output and `--error` for error logs
## Accounts and projects
Check your accounts at 
<a href="https://docs.alliancecan.ca/wiki/Frequently_Asked_Questions_about_the_CCDB#What_is_a_RAP.3F"> **Resources Allocation Project** </a>
Setting up your account information as environment variable
```
export SLURM_ACCOUNT=def-someuser
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT
```
##  Example of job scripts
### Serial jobs
A job requesting single score
```
#!/bin/bash
#SBATCH --time=<TIME> # resevering time spots
#SBATCH --account=def-<SOME_USER>
echo "Hello, world!"
sleep 30
```
### Array jobs
Submitting a whole set of jobns with one command
Each individual task is distingued by enviornment slurm variable `$SLURM_ARRAY_TASK_ID`, which is a range of values that can be set as `--array` params
 
 **Examples**
 ```
sbatch --array=0-7       # $SLURM_ARRAY_TASK_ID takes values from 0 to 7 inclusive
sbatch --array=1,3,5,7   # $SLURM_ARRAY_TASK_ID takes the listed values
sbatch --array=1-7:2     # Step size of 2, same as the previous example
sbatch --array=1-100%10  # Allows no more than 10 of the jobs to run simultaneously 
```
 
**Multiple parameters**
my_script_parallel.py
```
import time
import numpy as np
import sys

def calculation(x, beta):
    time.sleep(2) #simulate a long run
    return beta * np.linalg.norm(x**2)

if __name__ == "__main__":
    x = np.random.rand(100)
    betas = np.linspace(10,36.5,100) #subdivise the interval [10,36.5] with 100 values
    
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID (sys.argv[0] is the name of script, so params starts at 1)
    res = calculation(x,betas[i])
    print(res) #show the results on screen

# Run with: python my_script_parallel.py $SLURM_ARRAY_TASK_ID
```
data_parallel_python.sh
```
#!/bin/bash
#SBATCH --array=0-99
#SBATCH --time=1:00:00 (#SL
module load scipy-stack
python my_script_parallel.py $SLURM_ARRAY_TASK_ID
```
### Threaded jobs
```
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=0-0:5
#SBATCH --cpus-per-task=8 # Setting cpu core number
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./<Script_to_run>
```
### MPI jobs
```
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-00:05           # time (DD-HH:MM)
srun ./mpi_program               # mpirun or mpiexec also work
```
### GPU jobs
#### Requesting GPUs
`--gpus-per-node=[type:]number`
**Multi-threaded job**
```
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=1 # Number of GPU(s) per node
#SBATCH --cpus-per-task=6
#SBATCH --mem=400m #memory per node
#SBATCH --time=0-03:00
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
<Execute your program>
```
## Interactive jobs
Start interactive session using `salloc`
**Arguments**
- `--time=<TIME>` # Time to run 
- `--mem-per-cpu=<MEMORY>` # Allocate memory per cpu
- `--ntasks=<NUMBER>`# Specifies the number of tasks (processes) to run
- `--nodes=<NUMBER>` #Requests a specific number of nodes
- `--account=<ACCOUNT_NAME>` # Specify the account
- `--x11` # Enable graphical application

**Examples of running a python script**
```
salloc --time=2:00:00 --mem-per-cpu=8G --ntasks=4 --account=def-someuser
module load python/3.8 # Can Load more modules if required
python <PYTHON SCRIPT> # Can run other scripts using a different interpreter
exit
```
## Monitoring Jobs
### Current jobs
`squeue -u $USER` # Show all the jobs that is allocated to the user (Can use `sq` instead)

Can view specific jobs
```
squeue -u <USER> -t RUNNING
squeue -u <USER> _t PENDING
```
`$ scontrol show job -dd <jobid>`# View detailed information for specific jobs

### Email Notification
```
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=ALL
```
### Complated jobs
`seff <job_id>` # Short summary of completed job
`sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed` # Get detailed informatuon about the completed job

### Atattching to a runnign job

`srun --jobid <job_id> --pty watch -n Mseconds> nvidia-smi` # Display the GPU usage every interval
You can launch multiple monitoring commands using [`tmux`]
`$ srun --jobid <job_id> --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach`

## Cancelling jobs
To cancel a job:
`scancel <jobid>`
Cancel specific jobs
`scancel -u <USER>`
`scancel -t <PENDING/RUNNING> -u <USER>`

## Resubmitting jobs for long-running computations
### Restarting using job arrays 
Use `--array=1-100%1` # Divide the task and make the cpu runs 1 at a time

**Example Script**
```
#!/bin/bash
#SLURM --acount=def-someuser
#SLURM --cpus-per-task=1
#SLURM --time=0-10:00
#SLURM --mem=100m
#SLURM --array=1-10%1

echo "Current Directory $(pwd)"
echo "Current Time $(date)"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID . $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs"

if [ -e state.cpt ] 
# Check if there checkpoint file made
then
	mdrun --restart state.cpt
else
	mdrun
fi
echo "Job finisehd with exist code $? at $(date)"
```

### Resumbimssion from the job script

If the calculation is not finised and the time limit is over, the script sumbis a copy of itself to continue working

**Example**
```
#SBATCH --job-name=job_chain
#SBATCH --account=def-someuser
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mem=100M

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"

if [ -e state.cpt ]
then
	mdrun --restart state.cpt
else
	mdrun
fi
if <Condiction_to_check_if_job_finished>
then
	sbtach ${BASH_SOURCE[0]}
fi
echo "Job finished with exit code $? at: `date`"
```

# Creating virtual environment
Load python and other modules
`module load python/<version>`
`module load scipy-stack`

Create virtual environment
```
# Create a virtual environment for the project
virtualenv --no-download <PATH_TO_DIR_OF_ENV>
# Active the venv
source <ENV_DIR_PATH>/bin/activate
# Upgrade the pip in the environment and install requirements
pip install --no-index --upgrade pip (--no-idx tells to only from locally availale pakcages)
pip install --no-index -r requirements.txt
# Exit virtual environment
deactive
```

## Creaeting virtual environments inside jobs
```
module load StdEnv/2023 python/3.11 mpi4py

# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
EOF
# activate only on main node 
source $SLURM_TMPDIR/env/bin/activate;
# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python myscript-mpi.py;
```









