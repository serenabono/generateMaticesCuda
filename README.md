# generateMaticesCuda
Cuda to generate matrices of random noise 

> conda activate .py39env

> export LD_LIBRARY_PATH=/home/seb300/anaconda3/envs/.py39env/pkgs/cuda-toolkit/lib64:$LD_LIBRARY_PATH

To run:

> nvcc -o generatematrices generatematrices.cu -lcurand -lcublas

> ./generatematrices