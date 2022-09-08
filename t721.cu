/*
 * This program uses the host CURAND API to generate 10
 * pseudorandom doubles.  And then regenerate those same doubles.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void reduce(double *g_idata, double *g_odata);
void sum_array(double *a, size_t N);
void division(int N, double* A, int* B);

int main(int argc, char *argv[])
{
    size_t n = 10;
    size_t i;
    curandGenerator_t gen;
    double *devData, *hostData;

    /* Allocate n doubles on host */
    hostData = (double *)calloc(n, sizeof(double));

    /* Allocate n doubles on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                1234ULL));
    // generator offset = 0
    /* Generate n doubles on device */
    CURAND_CALL(curandGenerateNormalDouble(gen, devData, n, 0, 0.1));
    // generator offset = n
    /* Generate n doubles on device */
    CURAND_CALL(curandGenerateNormalDouble(gen, devData, n, 0, 0.1));
    // generator offset = 2n
    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double),
        cudaMemcpyDeviceToHost));

    
    /* Show result */
    for(int i = 0; i < n; i++) {
        printf("%.8f ", hostData[i]);
    }
    printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);
    
    return 1;
}


// void sum_array(double *a, size_t N){
//     double b[N]; // copies of a, b, c
//     double *dev_a, *dev_b; // device copies of a, b, c
//     double size = N * sizeof( double ); // we need space for 512 doubleegers

//     // allocate device copies of a, b, c
//     cudaMalloc( (void**)&dev_a, size );
//     cudaMalloc( (void**)&dev_b, size );

//     b[0] = 0;  //initialize the first value of b to zero
//     // copy inputs to device
//     cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
//     cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

//     dim3 blocksize(256); // create 1D threadblock
//     dim3 gridsize(N/blocksize.x);  //create 1D grid

//     //reduce<<<gridsize, blocksize>>>(dev_a, dev_b);

//     // copy device result back to host copy of c
//     cudaMemcpy( b, dev_b, sizeof( double ) , cudaMemcpyDeviceToHost );

//     printf("Reduced sum of Array elements = %f \n", b[0]);
//     cudaFree( dev_a );
//     cudaFree( dev_b );
// }

// __global__ void reduce(double *g_idata, double *g_odata) {

//     __shared__ double sdata[256];

//     // each thread loads one element from global to shared mem
//     // note use of 1D thread indices (only) in this kernel
//     int i = blockIdx.x*blockDim.x + threadIdx.x;

//     sdata[threadIdx.x] = g_idata[i];

//     __syncthreads();
//     // do reduction in shared mem
//     for (int s=1; s < blockDim.x; s *=2)
//     {
//         int index = 2 * s * threadIdx.x;;

//         if (index < blockDim.x)
//         {
//             sdata[index] += sdata[index + s];
//         }
//         __syncthreads();
//     }

//     // write result for this block to global mem
//     if (threadIdx.x == 0)
//         atomicAdd(g_odata,sdata[0]);
// }

// __global__ 
// void division(int N, double* A, double val)
// {
//     for(int row=blockIdx.x; row<N; row+=gridDim.x) {
//         for(int col=threadIdx.x; col<=row; col+=blockDim.x) {
//             A[row*N+col] /= (double)val;
//         }
//     }
// }