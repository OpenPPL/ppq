# include "linear.h"

__device__ 
inline int _round2int(
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

__device__ void generateRandomNumbers_wallace(  
    unsigned seed,  // Initialization seed  
    float *chi2Corrections,  // Set of correction values  
    float *globalPool,  // Input random number pool  
    float *output  // Output random numbers  


    unsigned tid=threadIdx.x;  
    // Load global pool into shared memory.  
    unsigned offset = __mul24(POOL_SIZE, blockIdx.x);  
    for( int i = 0; i < 4; i++ )  
        pool[tid+THREADS*i] = globalPool[offset+TOTAL_THREADS*i+tid];  
    __syncthreads();  
        const unsigned lcg_a=241;  
        const unsigned lcg_c=59;  
        const unsigned lcg_m=256;  
        const unsigned mod_mask = lcg_m-1;  
        seed=(seed+tid)&mod_mask ;  
        // Loop generating outputs repeatedly  
        for( int loop = 0; loop < OUTPUTS_PER_RUN; loop++ )  
        {  
        Transform();  
        unsigned intermediate_address;  
        i_a = __mul24(loop,8*TOTAL_THREADS)+8*THREADS *  
            blockIdx.x + threadIdx.x;  
        float chi2CorrAndScale=chi2Corrections[  
            blockIdx.x * OUTPUTS_PER_RUN + loop];  
        for( i = 0; i < 4; i++ )  
            output[i_a + i*THREADS]=chi2CorrAndScale*pool[tid+THREADS*i];  
        }  
    }

template<typename Dtype>
__global__ 
static void _gaussian_linear_tensor_quantize(
    const int num_of_elements,
    const Dtype* source,
    Dtype* dest,
    const float* scales,
    const int*   offsets,
    const int minimum,
    const int maximum,
    const int rounding
){
    float scale  = scales[0];
    int   offset = offsets[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_elements; i += blockDim.x * gridDim.x)
    {
        int quantized_value = _round2int(source[i] / scale, rounding) + offset;
        if(quantized_value > maximum) quantized_value = maximum;
        if(quantized_value < minimum) quantized_value = minimum;
        dest[i] = (quantized_value - offset) * scale;
    }
}

template<typename Dtype>
__global__ 
static void _gaussian_linear_channel_quantize(
    const int num_of_elements,
    const Dtype* source,
    Dtype* dest,
    const float* scales,
    const int*   offsets,
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

Tensor cuda_linear_tensor_gaussian_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    const int minimum,
    const int maximum,
    const int rounding,
    const float sigma
){
    Tensor dest = at::empty_like(source);
    const int num_of_elements = source.numel();
    const int num_of_blocks  = (num_of_elements + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    const float *scales_ptr  = scales.data_ptr<float>();
    const int   *offsets_ptr  = offsets.data_ptr<int>();
    
    if (source.dtype() == at::kFloat){
        float *dest_ptr   = dest.data_ptr<float>();
        const float *source_ptr = source.data_ptr<float>();
        _linear_tensor_quantize<float> <<<num_of_blocks, CUDA_NUM_THREADS>>>(
            num_of_elements, source_ptr, dest_ptr, scales_ptr, offsets_ptr, minimum, maximum, rounding
        );
    }
    else {
        throw "Unsupported tensor dtype.";
    }
    return dest;
}

Tensor cuda_linear_channel_gaussian_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding,
    const float sigma
){
    Tensor dest = at::empty_like(source);
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
    return dest;
}
