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

__global__ void _Histogram_T(
    const int64_t num_of_elements,
    const int64_t num_of_bins,
    const float* value,
    const float hist_scale,
    const bool clip_outliers,
    int* hist){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_elements){
        int b = floor(fabs(value[iter]) / hist_scale);
        if (clip_outliers && b > num_of_bins - 1) continue;
        else if (b > num_of_bins - 1) b = num_of_bins - 1;
        atomicAdd(&hist[b], 1); // fast enough ...
    }
}

__host__ void Histogram_T(
    const Tensor &value,
    const float hist_scale,
    const bool clip_outliers,
    Tensor &hist){
    /** 
     * PPQ Tensorwise Histogram Implementation
     * This function computes histogram of given value.
     * Result will sum up to hist tensor.
     * 
     * Say we have a float value f, and a float value hist_scale
     * We will select hist_bin = floor(f / hist_scale)
     */
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(hist, at::kInt, "Histogram(INT32)");

    _Histogram_T<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value)), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), NUM_OF_ELEMENT(hist), PTR<float>(value),
        hist_scale, clip_outliers, PTR<int>(hist)
    );
}

__global__ void _Histogram_C(
    const int64_t num_of_elements,
    const int64_t element_per_channel,
    const int     num_of_channel,
    const int64_t num_of_bins,
    const float*  value,
    const float   hist_scale,
    const bool    clip_outliers,
    int* hist){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_elements){
        int b = floor(fabs(value[iter]) / hist_scale);
        if (clip_outliers && b > num_of_bins - 1) continue;
        else if (b > num_of_bins - 1) b = num_of_bins - 1;

        int c = (iter / element_per_channel) % num_of_channel;
        atomicAdd(&hist[c * num_of_bins + b], 1); // fast enough ...
    }
}

__host__ void Histogram_C(
    const Tensor &value,
    const int channel_axis,
    const float hist_scale,
    const bool clip_outliers,
    Tensor &hist){
    /** 
     * PPQ Channelwise Histogram Implementation
     * This function computes histogram of given value along with given channel.
     * Result will sum up to hist tensor.
     * 
     * Say we have a float value f, and a float value hist_scale
     * We will select hist_bin = floor(f / hist_scale)
     */
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(hist, at::kInt, "Histogram(INT32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    if(NUM_OF_ELEMENT(hist) % num_of_channel != 0)
        throw InvalidValueException("Kernel Failure, Histogram shape is invalid.");

    _Histogram_C<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value)), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        int(NUM_OF_ELEMENT(hist) / num_of_channel), PTR<float>(value),
        hist_scale, clip_outliers, PTR<int>(hist)
    );
}