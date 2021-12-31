# include "sieve.h"

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
__global__ static void _linear_tensor_quant_sieve(
    const int num_of_elements,
    Dtype* tensor,
    Dtype* fp_offsets,
    Dtype* quantized,
    Dtype* mask,
    const float scale,
    const int offset,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        if(fp_offsets[i] > offset_limit) fp_offsets[i] = offset_limit;
        if(fp_offsets[i] < -offset_limit) fp_offsets[i] = -offset_limit;

        float value = _round2int(tensor[i] / scale + fp_offsets[i], rounding) + offset;
        if(value > maximum) value = maximum;
        if(value < minimum) value = minimum;
        quantized[i] = (value - offset) * scale;

        float diff = abs(tensor[i] - quantized[i]) / scale;
        mask[i] = diff < 1 - threshold;
    }
}

template<typename Dtype>
__global__ static void _linear_channel_quant_sieve(
    const int num_of_elements,
    Dtype* tensor,
    Dtype* fp_offsets,
    Dtype* quantized,
    Dtype* mask,
    float* scales,
    int* offsets,
    const int element_per_channel,
    const int num_of_channel,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        if(fp_offsets[i] > offset_limit) fp_offsets[i] = offset_limit;
        if(fp_offsets[i] < -offset_limit) fp_offsets[i] = -offset_limit;
        
        int channel = (i / element_per_channel) % num_of_channel;
        float value = _round2int(tensor[i] / scales[channel] + fp_offsets[i], rounding) + offsets[channel];
        if(value > maximum) value = maximum;
        if(value < minimum) value = minimum;
        quantized[i] = (value - offsets[channel]) * scales[channel];
        
        float diff = abs(tensor[i] - quantized[i]) / scales[channel];
        mask[i] = diff < 1 - threshold;
    }
}

vector<Tensor> cuda_linear_tensor_quant_sieve(
    const Tensor &tensor,
    const Tensor &fp_offset_tensor,
    const float scale,
    const int offset,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = tensor.numel();
    const int num_of_blocks  = (num_of_elements + num_of_threads - 1) / num_of_threads;

    if (tensor.dtype() == at::kFloat){
        float *tensor_ptr    = tensor.data_ptr<float>();
        float *fp_offset_ptr = fp_offset_tensor.data_ptr<float>();
        
        Tensor quantized     = at::empty_like(tensor);
        Tensor mask          = at::empty_like(tensor);

        float *quantized_ptr = quantized.data_ptr<float>();
        float *mask_ptr      = mask.data_ptr<float>();

        _linear_tensor_quant_sieve<float> <<<num_of_blocks, num_of_threads>>>(
            num_of_elements, tensor_ptr, fp_offset_ptr,
            quantized_ptr, mask_ptr, scale, offset, minimum, maximum, 
            rounding, offset_limit, threshold
        );

        return  {quantized, mask == 1};
    }
    else {
        throw "Unsupported tensor dtype.";
    }
}

vector<Tensor> cuda_linear_channel_quant_sieve(
    const Tensor &tensor,
    const Tensor &fp_offset_tensor,
    const Tensor &scales,
    const Tensor &offsets,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
){
    constexpr int num_of_threads = CUDA_NUM_THREADS;
    const int num_of_elements = tensor.numel();
    const int num_of_blocks  = (num_of_elements + num_of_threads - 1) / num_of_threads;
    
    // calculate element_per_channel, num_of_channel
    int element_per_channel = 1;
    const int num_of_channel = tensor.sizes()[channel_axis];
    for(int axis = tensor.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= tensor.sizes()[axis];
    }
    
    float *scale_ptr = scales.data_ptr<float>();
    int *offset_ptr = offsets.data_ptr<int>();

    if (tensor.dtype() == at::kFloat){

        float *tensor_ptr    = tensor.data_ptr<float>();
        float *fp_offset_ptr = fp_offset_tensor.data_ptr<float>();
        
        Tensor quantized     = at::empty_like(tensor);
        Tensor mask          = at::empty_like(tensor);

        float *quantized_ptr = quantized.data_ptr<float>();
        float *mask_ptr      = mask.data_ptr<float>();

        _linear_channel_quant_sieve<float> <<<num_of_blocks, num_of_threads>>>(
            num_of_elements, tensor_ptr, fp_offset_ptr, quantized_ptr,
            mask_ptr, scale_ptr, offset_ptr, element_per_channel, num_of_channel, 
            minimum, maximum, rounding, offset_limit, threshold
        );
        
        return {quantized, mask == 1};
    }
    else {
        throw "Unsupported tensor dtype.";
    }
}
