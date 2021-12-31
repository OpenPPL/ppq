# include "linear.h"

__device__ inline int _round2int(
    const float value, 
    const int rounding
){
    switch(rounding){
        case ROUND_HALF_EVEN:
            return __float2int_rn(value);
        case ROUND_HALF_UP:
            return floor(value + .5);
        case ROUND_HALF_DOWN:
            return ceil(value - .5);
        case ROUND_HALF_TOWARDS_ZERO:
            if (value > 0) return _round2int(value, ROUND_HALF_DOWN);
            else return _round2int(value, ROUND_HALF_UP);
        case ROUND_HALF_FAR_FORM_ZERO:
            if (value > 0) return _round2int(value, ROUND_HALF_UP);
            else return _round2int(value, ROUND_HALF_DOWN);
        case ROUND_UP:
            return ceil(value);
        case ROUND_DOWN:
            return floor(value);
        default:
            return __float2int_rn(value);
    }
    return 0;
}

template<typename Dtype>
__global__ static void _linear_tensor_quantize(
    const int num_of_elements,
    Dtype* source,
    Dtype* dest,
    const float scale,
    const int offset,
    const int minimum,
    const int maximum,
    const int rounding
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        int quantized_value = _round2int(source[i] / scale, rounding) + offset;
        if(quantized_value > maximum) quantized_value = maximum;
        if(quantized_value < minimum) quantized_value = minimum;
        dest[i] = (quantized_value - offset) * scale;
    }
}

template<typename Dtype>
__global__ static void _linear_channel_quantize(
    const int num_of_elements,
    Dtype* source,
    Dtype* dest,
    float* scales,
    int* offsets,
    const int element_per_channel,
    const int num_of_channel,
    const int minimum,
    const int maximum,
    const int rounding
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        int channel = (i / element_per_channel) % num_of_channel;
        int quantized_value = _round2int(source[i] / scales[channel], rounding) + offsets[channel];
        if(quantized_value > maximum) quantized_value = maximum;
        if(quantized_value < minimum) quantized_value = minimum;
        dest[i] = (quantized_value - offsets[channel]) * scales[channel];
    }
}

template<typename Dtype>
__global__ static void _tensor_histogram(
    const int num_of_elements,
    const int num_of_bins,
    Dtype* source,
    int* dest,
    const float scale,
    const int offset,
    const bool abs,
    const int rounding
){
    extern __shared__ int histogram[];
    for (int i = threadIdx.x; i < num_of_bins; i += CUDA_NUM_THREADS){
        histogram[i] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        int selected_bin;
        if(abs && source[i] < 0) selected_bin = __float2int_rz(- source[i] / scale) + offset;
        else selected_bin = floor(source[i] / scale) + offset;

        selected_bin = selected_bin >= 0 ? selected_bin : 0;
        selected_bin = selected_bin < num_of_bins ? selected_bin : num_of_bins - 1;
        atomicAdd(&histogram[selected_bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_of_bins; i += CUDA_NUM_THREADS){
        atomicAdd(&dest[i], histogram[i]);
    }
}

/*
template<typename Dtype>
__global__ static void _mse_quantize_scale_searching(
    const int num_of_elements,
    const float* scales,
    const int *offsets,
    Dtype* source,
    Dtype* dest,
    const int minimum,
    const int maximum,
    const int rounding,
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        int quantized_value = _round2int(source[i] / scale, rounding) + offset;
        if(quantized_value > maximum) quantized_value = maximum;
        if(quantized_value < minimum) quantized_value = minimum;
        dest[i] = (quantized_value - offset) * scale;
    }
}
*/

void cuda_linear_tensor_quantize(
    const Tensor &source,
    Tensor &dest,
    const float scale,
    const float offset,
    const int minimum,
    const int maximum,
    const int rounding
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_blocks  = (num_of_elements + num_of_threads - 1) / num_of_threads;

    if (source.dtype() == at::kFloat){
        float *source_data = source.data_ptr<float>();
        float *dest_data   = dest.data_ptr<float>();
        _linear_tensor_quantize<float> <<<num_of_blocks, num_of_threads>>>(
            num_of_elements, source_data, dest_data, scale, offset, minimum, maximum, rounding
        );
    }
    else {
        throw "Unsupported tensor dtype.";
    }
}

void cuda_linear_channel_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    Tensor &dest,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_blocks  = (num_of_elements + num_of_threads - 1) / num_of_threads;
    
    int element_per_channel = 1;
    const int num_of_channel = source.sizes()[channel_axis];
    for(int axis = source.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= source.sizes()[axis];
    }
    
    float *scale_data_ptr = scales.data_ptr<float>();
    int *offset_data_ptr = offsets.data_ptr<int>();

    // calculate element_per_channel, num_of_channel
    if (source.dtype() == at::kFloat){
        float *source_data = source.data_ptr<float>();
        float *dest_data   = dest.data_ptr<float>();
        _linear_channel_quantize<float> <<<num_of_blocks, num_of_threads>>>(
            num_of_elements, source_data, dest_data, scale_data_ptr, offset_data_ptr, 
            element_per_channel, num_of_channel, minimum, maximum, rounding
        );
    }
    else {
        throw "Unsupported tensor dtype.";
    }
}

void cuda_tensor_histogram(
    const Tensor &source,
    Tensor &dest,
    const float scale,
    const int offset,
    const bool abs,
    const int rounding
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = source.numel();
    const int num_of_bins     = dest.numel();
    const int num_of_blocks   = (num_of_elements + num_of_threads - 1) / num_of_threads;

    if (abs && offset != 0) throw "offset is invalid when abs mode is activated, please set it to 0.";
    if (source.dtype() == at::kFloat){
        float *source_data = source.data_ptr<float>();
        int *dest_data   = dest.data_ptr<int>();
        _tensor_histogram<float> <<<num_of_blocks, num_of_threads, sizeof(int) * num_of_bins>>>(
            num_of_elements, num_of_bins, source_data, dest_data, scale, offset, abs, rounding
        );
    }
    else {
        throw "Unsupported tensor dtype.";
    }
}