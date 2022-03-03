# include "sort.h"
# include <thrust/sort.h>
# include <thrust/execution_policy.h>


__global__ 
static void _Quantile_T(
    const float *source,
    float *dest,
    const int64_t num_of_elements,
    const float q
){
    int max_pos = __float2int_rn(num_of_elements * q);
    max_pos = CLIP<int>(max_pos, 0, num_of_elements - 1);
    dest[0] = source[max_pos];

    int min_pos = __float2int_rn(num_of_elements * (1 - q));
    min_pos = CLIP<int>(min_pos, 0, num_of_elements - 1);
    dest[1] = source[min_pos];
}


__global__ 
static void _cuda_order_preserving_observe(
    const float *source,
    float *dest,
    const int64_t num_of_elements
){
    if(num_of_elements == 1){
        dest[0] = source[0];
        dest[1] = source[0];
        dest[2] = source[0];
        dest[3] = source[0];
    } else {
        dest[0] = source[num_of_elements - 1];
        dest[1] = source[num_of_elements - 2];
        dest[2] = source[0];
        dest[3] = source[1];
    }
}

__host__ Tensor Quantile_T(const Tensor &source, const float q){
    CheckTensor(source, at::kFloat, "Value(FP32)");
    const int64_t num_of_elements = NUM_OF_ELEMENT(source);
    
    Tensor dest = at::empty({2}, source.options());
    Tensor value = source.clone();

    thrust::sort(
        thrust::device, 
        PTR<float>(value),
        PTR<float>(value) + num_of_elements);

    _Quantile_T<<<1, 1>>>(
        PTR<float>(value), 
        PTR<float>(dest), 
        num_of_elements, q);
    return dest;
}


Tensor cuda_order_preserving_observe(Tensor &source){
    const int num_of_elements = source.numel();
    
    Tensor dest = at::empty({4}, source.options());
    Tensor source_cpy = source.clone();

    float *source_ptr = source_cpy.data_ptr<float>();
    float *dest_ptr   = dest.data_ptr<float>();

    thrust::sort(thrust::device, source_ptr, source_ptr + num_of_elements);
    _cuda_order_preserving_observe<<<1, 1>>>(source_ptr, dest_ptr, num_of_elements);
    return dest;
}
