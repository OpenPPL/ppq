# include "sort.h"
# include <thrust/sort.h>
# include <thrust/execution_policy.h>


__global__ static void _cuda_quantile(
    const float *source,
    float *dest,
    const int num_of_elements,
    float q
){
    int max_pos = __float2int_rn(num_of_elements * q);
    if (max_pos >= num_of_elements) max_pos = num_of_elements - 1;
    if (max_pos <= 0) max_pos = 0;
    dest[0] = source[max_pos];

    int min_pos = __float2int_rn(num_of_elements * (1 - q));
    if (min_pos >= num_of_elements) min_pos = num_of_elements - 1;
    if (min_pos <= 0) max_pos = 0;
    dest[1] = source[min_pos];
}

Tensor cuda_quantile(Tensor &source, const float q){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    
    Tensor dest = at::empty({2}, source.options());
    Tensor source_cpy = source.clone();

    float *source_ptr = source_cpy.data_ptr<float>();
    float *dest_ptr   = dest.data_ptr<float>();

    thrust::sort(thrust::device, source_ptr, source_ptr + num_of_elements);
    _cuda_quantile<<<1, 1>>>(source_ptr, dest_ptr, num_of_elements, q);
    return dest;
}