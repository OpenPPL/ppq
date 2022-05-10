# include "test.h"

__device__ inline int _round2int(
    const float value,
    const int rounding
){
    switch(rounding){
        case ROUND_TO_NEAR_EVEN:
            return __float2int_rn(value);
        case ROUND_UP:
            return __float2int_ru(value);
        case ROUND_DOWN:
            return __float2int_rd(value);
        case ROUND_TO_ZERO:
            return __float2int_rz(value);
        default:
            return (int)value;
    }
    return 0;
}

template<typename Dtype>
__device__ static inline int _inline_int8_scale_relay(
    const float in_scale,
    const float out_scale,
    Dtype in_value){
    int cliped_value = _round2int(float(in_value * in_scale / out_scale), ROUND_TO_NEAR_EVEN);
    cliped_value = cliped_value > 127 ? 127 : cliped_value;
    cliped_value = cliped_value < -128 ? -128 : cliped_value;
    return int(cliped_value);
}

__global__ static void _kernel_int8_scale_relay(
    const int num_of_elements,
    int* source,
    int* dest,
    float in_scale,
    float out_scale
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x){
        dest[i] = _inline_int8_scale_relay(source[i], in_scale, out_scale);
    }
}

__global__ static void _dummy_pooling_v1(
    const int num_of_elements,
    const int* source,
    int* dest
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += 4 * blockDim.x * gridDim.x)
        dest[i] = int((source[i] + source[i + 1] + source[i + 2] + source[i + 3]) / 4.0f);
}

__global__ static void _dummy_pooling_v2(
    const int num_of_elements,
    const int* source,
    int* dest
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += 4 * blockDim.x * gridDim.x)
        dest[i] = int((source[i] + source[i + 1] + source[i + 2] + source[i + 3]) / 4);
}

__global__ static void _dummy_pooling_v3(
    const int num_of_elements,
    const float in_scale,
    const float out_scale,
    const int* source,
    int* dest
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += 4 * blockDim.x * gridDim.x){
        dest[i] = int((source[i] + source[i + 1] + source[i + 2] + source[i + 3]) / 4); // line alpha
        dest[i] = _inline_int8_scale_relay(in_scale, out_scale, dest[i]);               // line beta
    }
}

void dummy_pooling_v3(
    const Tensor source,
    Tensor dest,
    const float in_scale,
    const float out_scale
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_blocks   = (num_of_elements + num_of_threads - 1) / num_of_threads;

    int *source_data = source.data_ptr<int>();
    int *dest_data   = dest.data_ptr<int>();
    _dummy_pooling_v3 <<<num_of_blocks, num_of_threads>>>(
        num_of_elements, in_scale, out_scale, source_data, dest_data
    );
}

void dummy_pooling_v2(
    const Tensor source,
    Tensor dest,
    const float in_scale,
    const float out_scale
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_blocks   = (num_of_elements + num_of_threads - 1) / num_of_threads;

    int *source_data = source.data_ptr<int>();
    int *dest_data   = dest.data_ptr<int>();

    _dummy_pooling_v2 <<<num_of_blocks, num_of_threads>>>(
        num_of_elements, source_data, dest_data
    );
    _kernel_int8_scale_relay <<<num_of_blocks, num_of_threads>>>(
        num_of_elements, dest_data, dest_data, in_scale, out_scale
    );
}

void dummy_pooling_v1(
    const Tensor source,
    Tensor dest,
    const float in_scale,
    const float out_scale
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_blocks   = (num_of_elements + num_of_threads - 1) / num_of_threads;

    int *source_data = source.data_ptr<int>();
    int *dest_data   = dest.data_ptr<int>();

    _dummy_pooling_v1 <<<num_of_blocks, num_of_threads>>>(
        num_of_elements, source_data, dest_data
    );
    _kernel_int8_scale_relay <<<num_of_blocks, num_of_threads>>>(
        num_of_elements, dest_data, dest_data, in_scale, out_scale
    );
}
