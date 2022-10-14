# include "floating.h"
# include "common.cuh"

template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                            int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += WARP_SHFL_XOR(val, mask, 32, FINAL_MASK);
    return val;
}

template<typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduceSum(val);
    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (lane < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduceSum(val);
    return val;
}

__global__ void _QuantizeTensor_FT(
    const int64_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        exponent, 
    const int        mantissa,
    const float      clip_min,
    const float      clip_max,
    const Rounding   rounding,
    float* out
){
    float s = scale[0]; float o = offset[0]; int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        float qt = QuantizeScalarFloating<float, float, float>(
            __ldg(&value[iter]), s, o, exponent, mantissa, clip_min, clip_max, rounding);
        float deq = DequantizeScalar<float, float, float>(qt, s, o);
        out[iter] = deq;
    }
}

__host__ Tensor QuantizeTensor_FT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, const Rounding rounding){
    /**
     * PPQ Tensor Quantization Function implementation.
     * This function quantizes a float tensor(tensor wise) to low-precision float
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    Tensor quantized = at::empty_like(value);
    _QuantizeTensor_FT<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        exponent, mantissa, clip_min, clip_max, rounding, PTR<float>(quantized)
    );
    return quantized;
}

__global__ void _QuantizeTensor_FC(
    const int64_t    num_of_element,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        exponent, 
    const int        mantissa,
    const float      clip_min,
    const float      clip_max,
    const Rounding   rounding,
    float* out
){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        int c = (iter / element_per_channel) % num_of_channel;
        auto qt = QuantizeScalarFloating<float, float, float>(
            __ldg(&value[iter]), __ldg(&scale[c]), __ldg(&offset[c]),
            exponent, mantissa, clip_min, clip_max, rounding);
        float deq = DequantizeScalar<float, float, float>(qt, scale[c], offset[c]);
        out[iter] = deq;
    }
}

__host__ Tensor QuantizeTensor_FC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, const int channel_axis,
    const Rounding rounding){
    /**
     * PPQ Tensor Quantization Function implementation.
     * This function quantizes a float tensor(tensor wise) to low-precision float
     *
     * This function is channel wise quantization function
     * Each channel has its own scale and offset.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    Tensor quantized = at::empty_like(value);
    _QuantizeTensor_FC<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        exponent, mantissa, clip_min, clip_max, rounding, PTR<float>(quantized)
    );
    return quantized;
}

__global__
void _QuantizeTensor_FT_B(
    const int64_t num_of_elements,
    const float* value,
    const float* scales,
    const float* offsets,
    const float* grad_y,
    const int exponent, 
    const int mantissa,
    const float clip_min,
    const float clip_max,
    const Rounding rounding,
    float* grad_s,
    float* grad_x
){
    int64_t iter; 
    float s = scales[0];
    float inv_s = 1 / s; 
    float o = offsets[0];
    float _clip_min = s * (clip_min - o);
    float _clip_max = s * (clip_max - o);

    KERNEL_LOOP(iter, num_of_elements){
        float v  = __ldg(&value[iter]);
        float dy = __ldg(&grad_y[iter]);

        float qt = QuantizeScalarFloating<float, float, float>(
            v, s, o, exponent, mantissa, clip_min - 1, clip_max + 1, rounding);
        float q = DequantizeScalar<float, float, float>(qt, s, o);

        /* Calculate grad for scale and value */
        float partial_gard_s = 0;
        if(qt == clip_max + 1) {
            partial_gard_s = _clip_max * dy * inv_s;
            grad_x[iter] = 0;
        }
        else if(qt == clip_min - 1) {
            partial_gard_s = _clip_min * dy * inv_s;
            grad_x[iter] = 0;
        } else {
            partial_gard_s = (q - v) * inv_s * dy;
            grad_x[iter] = dy;
        }

        /* Reduce Gradient */
        __syncthreads();
        float reduced_grad_s = BlockReduceSum<float>(partial_gard_s);
        if (threadIdx.x == 0) {
            atomicAdd(grad_s, reduced_grad_s / sqrtf((float)(num_of_elements * clip_max)));
        }
    }
}

__host__ std::vector<Tensor> QuantizeTensor_FT_B(
    const Tensor &value, const Tensor &scales, const Tensor &offsets, const Tensor &grad_y,
    const int exponent, const int mantissa, const float clip_min, const float clip_max, 
    const Rounding rounding
){
    /**
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scales, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offsets, at::kFloat, "Offset(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");

    constexpr int NUM_OF_THREADS = 1024;
    Tensor grad_s = at::zeros_like(scales);
    Tensor grad_x = at::zeros_like(grad_y);

    _QuantizeTensor_FT_B<<<
        NUM_OF_BLOCK(NUM_OF_ELEMENT(value), NUM_OF_THREADS), 
        NUM_OF_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        PTR<float>(value),
        PTR<float>(scales),
        PTR<float>(offsets),
        PTR<float>(grad_y),
        exponent,
        mantissa,
        clip_min,
        clip_max,
        rounding,
        PTR<float>(grad_s),
        PTR<float>(grad_x)
    );
    return {grad_x, grad_s};
}

__global__
void _QuantizeTensor_FC_B(
    const int64_t num_of_elements,
    const int64_t element_per_channel,
    const int     num_of_channel,
    const float*  value,
    const float*  scales,
    const float*  offsets,
    float*        grad_y,
    const int     exponent, 
    const int     mantissa,
    const float   clip_min,
    const float   clip_max,
    const Rounding rounding,
    float *       grad_s,
    float *       grad_x
){
    int channel_idx = blockIdx.x;
    float s = scales[channel_idx]; float inv_s = 1 / s; 
    float o = offsets[channel_idx];
    float _clip_min = s * (clip_min - o);
    float _clip_max = s * (clip_max - o);
    float partial_gard_s = 0;

    for(int64_t iter = (blockIdx.x * element_per_channel) + blockIdx.y * CUDA_NUM_THREADS + threadIdx.x;
        iter < num_of_elements; iter += element_per_channel * num_of_channel)
    {
        // if processing element is not belongs to correct channel, skip its computation
        if (blockIdx.y * CUDA_NUM_THREADS + threadIdx.x < element_per_channel){
            float dy = __ldg(&grad_y[iter]);
            float v  = __ldg(&value[iter]);

            float qt = QuantizeScalarFloating<float, float, float>(
                v, s, o, exponent, mantissa, clip_min - 1, clip_max + 1, rounding);
            float q = DequantizeScalar<float, float, float>(qt, s, o);

            /* Calculate grad for scale and value */
            if(qt == clip_max + 1) {
                partial_gard_s = _clip_max * dy * inv_s;
                grad_x[iter] = 0;
            }
            else if(qt == clip_min - 1) {
                partial_gard_s = _clip_min * dy * inv_s;
                grad_x[iter] = 0;
            } else {
                partial_gard_s = (q - v) * inv_s * dy;
                grad_x[iter] = dy;
            }
        }
        else{
            partial_gard_s = 0;
        }

        /* Reduce Gradient */
        __syncthreads();
        float reduced_grad_s = BlockReduceSum<float>(partial_gard_s);
        if (threadIdx.x == 0) {
            atomicAdd(&grad_s[channel_idx], reduced_grad_s / sqrtf((float)(num_of_elements * clip_max)));
        }
    }
}


__host__ std::vector<Tensor> QuantizeTensor_FC_B(
    const Tensor &value, const Tensor &scales, const Tensor &offsets, 
    const Tensor &grad_y, const int exponent, const int mantissa,
    const float clip_min, const float clip_max,
    const Rounding rounding, const int channel_axis
){
    /**
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scales, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offsets, at::kFloat, "Offset(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");

    Tensor grad_s = at::zeros_like(scales);
    Tensor grad_x = at::zeros_like(grad_y);
    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }
    constexpr int NUM_OF_THREADS = 1024;
    dim3 grid;
    grid.x = static_cast<unsigned int>(num_of_channel);
    grid.y = static_cast<unsigned int>(NUM_OF_BLOCK(element_per_channel, NUM_OF_THREADS));
    grid.z = 1;

    _QuantizeTensor_FC_B<<<grid, NUM_OF_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        element_per_channel,
        num_of_channel,
        PTR<float>(value),
        PTR<float>(scales),
        PTR<float>(offsets),
        PTR<float>(grad_y),
        exponent,
        mantissa,
        clip_min,
        clip_max,
        rounding,
        PTR<float>(grad_s),
        PTR<float>(grad_x)
    );
    return {grad_x, grad_s};
}
