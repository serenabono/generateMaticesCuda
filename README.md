# generateMaticesCuda
Cuda to generate matrices of random noise 

To run:

> nvcc -o generatematrices generatematrices.cu -lcurand -lcublas

> ./generatematrices