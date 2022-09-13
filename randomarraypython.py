import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
code = """
    #include <curand_kernel.h>
    const int nstates = %(NGENERATORS)s;
    __device__ curandState_t* states[nstates];
    extern "C" {
        __global__ void initkernel(int seed)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tidx < nstates) {
                curandState_t* s = new curandState_t;
                if (s != 0) {
                    curand_init(seed, tidx, 0, s);
                }
                states[tidx] = s;
            }
        }
        __global__ void randfillkernel(float *values, int N)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tidx < nstates) {
                curandState_t s = *states[tidx];
                for(int i=tidx; i < N; i += blockDim.x * gridDim.x) {
                    values[i] = abs(curand_normal(&s));
                }
                *states[tidx] = s;
            }
        }
    }
"""
import time
import random
from pycuda.elementwise import ElementwiseKernel
N = 1024
mod = SourceModule(code % { "NGENERATORS" : N }, no_extern_c=True)
init_func = mod.get_function("initkernel")
fill_func = mod.get_function("randfillkernel")
start_cuda = time.time()
seed = np.int32(123456789)
nvalues = 3111696
init_func(seed, block=(N,1,1), grid=(1,1,1))
gdata = gpuarray.zeros(nvalues, dtype=np.float32)
fill_func(gdata, np.int32(nvalues), block=(N,1,1), grid=(1,1,1))
gdata_sumrows = pycuda.gpuarray.sum(gdata)
c_gpu = gpuarray.empty_like(gdata)
lin_comb = ElementwiseKernel(
        "float a, float *x, float *z",
        "z[i] = a*x[i]",
        "linear_combination")
lin_comb(1/gdata_sumrows.get(), gdata, c_gpu)
end_cuda = time.time()


start_cuda = time.time()
seed = np.int32(123456789)
nvalues = 3111696
init_func(seed, block=(N,1,1), grid=(1,1,1))
gdata = gpuarray.zeros(nvalues, dtype=np.float32)
fill_func(gdata, np.int32(nvalues), block=(N,1,1), grid=(1,1,1))
gdata_sumrows = pycuda.gpuarray.sum(gdata)
c_gpu = gpuarray.empty_like(gdata)
lin_comb = ElementwiseKernel(
        "float a, float *x, float *z",
        "z[i] = a*x[i]",
        "linear_combination")
lin_comb(1/gdata_sumrows.get(), gdata, c_gpu)
end_cuda = time.time()