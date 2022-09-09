#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <math.h>
#include <fstream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CUBLA_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

void matTofile(int m, int n, curandGenerator_t rng);

struct Abs: public thrust::unary_function<double, double>
{
    __host__ __device__ double operator()(double x)
    {
        return abs(x);
    }
};

struct Inv: public thrust::unary_function<double, double>
{
    __host__ __device__ double operator()(double x)
    {
        return (double) 1.0 / x;
    }
};

template<typename T>
struct MulC: public thrust::unary_function<T, T>
{
    T C;
    __host__ __device__ MulC(T c) :
        C(c)
    {
    }
    __host__ __device__ T operator()(T x)
    {
        return x * C;
    }
};

template<typename T>
struct line2col: public thrust::unary_function<T, T>
{
    T C;
    __host__ __device__ line2col(T C) :
            C(C)
    {
    }

    __host__ __device__ T operator()(T i)
    {
        return i / C;
    }
};

int main()
{
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasHandle_t hd;
    curandGenerator_t rng;
    cublasCreate(&hd);
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    const size_t m =19600, n = 19600;

    int times = 214200 / 80;

    float size = m*n*5 * sizeof( float );
    std::cout << "size: " << size << ", times: " << times <<std::endl;
    std::cout << time(NULL) << std::endl;
    for(int i = 0; i < times; ++i){
        matTofile(m,n,rng);
    }
    std::cout << time(NULL) << std::endl;
    curandDestroyGenerator(rng);
    cublasDestroy(hd);

    return 0;
}

void matTofile(int m, int n, curandGenerator_t rng){
    std::ofstream ofile("rands_out.txt");
    thrust::device_vector<double> A(m * n);
    thrust::device_vector<double> C(m * n);
    thrust::device_vector<double> sum1(1 * n);
    thrust::device_vector<double> sum2(1 * n);
    thrust::device_vector<double> one(m * n, 1);

    double* pA = thrust::raw_pointer_cast(&A[0]);
    double* pSum1 = thrust::raw_pointer_cast(&sum1[0]);
    double* pSum2 = thrust::raw_pointer_cast(&sum2[0]);
    double* pOne = thrust::raw_pointer_cast(&one[0]);

    curandGenerateUniformDouble(rng, pA, A.size());

    const int count = 2;
    
    for (int i = 0; i < count; i++)
    {
        thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), line2col<int>(m)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), line2col<int>(m)) + A.size(),
                thrust::make_transform_iterator(A.begin(), Abs()),
                thrust::make_discard_iterator(),
                sum2.begin());
        thrust::transform(
                A.begin(), A.end(),
                thrust::make_permutation_iterator(
                        sum2.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), line2col<int>(m))),
                C.begin(),
                thrust::divides<double>());
    }

    // thrust::copy(C.begin(), C.end(), std::ostream_iterator<double>(ofile, ","));

}