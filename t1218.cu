#include <stdio.h>

#define w 64
#define h 64
#define N w*h

__global__ void reduce(float *g_idata, float *g_odata);
void fill_array (float *a, float n);

int main( void ) {
    float a[N], b[N]; // copies of a, b, c
    float *dev_a, *dev_b; // device copies of a, b, c
    float size = N * sizeof( float ); // we need space for 512 floategers

    // allocate device copies of a, b, c
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );

    fill_array( a, N );
    b[0] = 0;  //initialize the first value of b to zero
    // copy inputs to device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    dim3 blocksize(256); // create 1D threadblock
    dim3 gridsize(N/blocksize.x);  //create 1D grid

    reduce<<<gridsize, blocksize>>>(dev_a, dev_b);

    // copy device result back to host copy of c
    cudaMemcpy( b, dev_b, sizeof( float ) , cudaMemcpyDeviceToHost );

    printf("Reduced sum of Array elements = %f \n", b[0]);
    printf("Value should be: %d \n", ((N-1)*(N/2)));
    cudaFree( dev_a );
    cudaFree( dev_b );

    return 0;
}

__global__ void reduce(float *g_idata, float *g_odata) {

    __shared__ float sdata[256];

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = g_idata[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
        atomicAdd(g_odata,sdata[0]);
}

// CPU function to generate a vector of random floategers
void fill_array (float *a, float n)
{
    for (int i = 0; i < n; i++)
        a[i] = i;
}