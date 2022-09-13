# generateMaticesCuda
Cuda to generate matrices of random noise 

> conda activate .py39env

> conda install -c conda-forge cudatoolkit-dev

> export LD_LIBRARY_PATH=/home/seb300/anaconda3/envs/.py39env/pkgs/cuda-toolkit/lib64:$LD_LIBRARY_PATH

To run:

> nvcc --std=c++11 -DNDEBUG -lcurand -lcublas -o generatematrices generatematrices.cu 

> ./generatematrices