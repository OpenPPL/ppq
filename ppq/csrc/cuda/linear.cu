# include "linear.h"
# include "common.cuh"
# include <chrono>
using namespace std::chrono;

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

template<int TPB>
__global__ void _QuantizeTensor_LT(
    const int32_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* out
){
    const int32_t index_offset = blockIdx.x * TPB + threadIdx.x;
    if (index_offset < num_of_element){
        float s = __ldg(&scale[0]); 
        int   o = std::round(__ldg(&offset[0]));
        float qt = QuantizeScalar<float, float, int>(
            __ldg(&value[index_offset]), s, o, clip_min, clip_max, rounding);
        float deq = DequantizeScalar<int, float, int>(qt, s, o);
        out[index_offset] = deq;
    }
}

template<int VPT, int TPB>
__global__ void _QuantizeTensorVectorize_LT(
    const int32_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float*           out
){
    const int32_t index_offset = blockIdx.x * VPT * TPB + threadIdx.x * VPT;
    if (index_offset < num_of_element){
        // never do (1 / s) here
        // value / s not always == value * (1 / s)
        float s = __ldg(&scale[0]);
        int o = std::round(__ldg(&offset[0]));

        # pragma unroll
        for(int32_t i = 0; i < VPT; i++){
            float qt = __ldg(&value[index_offset + i]) / s;
            int dq = CLIP<int>(_round2int(qt, rounding) + o, clip_min, clip_max);
            out[index_offset + i] = (dq - o) * s;
        }

    }
}

__host__ Tensor QuantizeTensor_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max, const Rounding rounding){
    /**
     * PPQ Tensor Quantization Function implementation.
     * This function quantizes a float tensor(tensor wise),
     * giving an quantized tensor as output
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = (clip(i / s + o) - o) * s
     * Where s is scale factor, and o is offset
     *
     * This function is tensor wise quantization function
     * All tensor share a same scale and offset
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");
    auto v_contiguous = value.contiguous();
    auto num_of_elements = value.numel();
    if(num_of_elements > 0x7fffffff) 
        throw InvalidValueException("There are too many element in your tensor(more than 2*10^9)"); 

    Tensor quantized = at::empty_like(v_contiguous);
    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 4;

    if(num_of_elements % VPT == 0){
        _QuantizeTensorVectorize_LT<VPT, TPB>
        <<<NUM_OF_BLOCK_NOLIMIT(num_of_elements, VPT * TPB), TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_of_elements, PTR<float>(v_contiguous), PTR<float>(scale), PTR<float>(offset),
            clip_min, clip_max, rounding, PTR<float>(quantized)
        );
    }
    else{
        _QuantizeTensor_LT<TPB>
        <<<NUM_OF_BLOCK_NOLIMIT(num_of_elements, TPB), TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_of_elements, PTR<float>(v_contiguous), PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, PTR<float>(quantized));
    }

    return quantized;
}

template<int TPB>
__global__ void _QuantizeTensor_LC(
    const int32_t    num_of_element,
    const int32_t    element_per_channel,
    const int32_t    num_of_channel,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* out
){
    const int32_t index_offset = blockIdx.x * TPB + threadIdx.x;
    if (index_offset < num_of_element){
        int c = (index_offset / element_per_channel) % num_of_channel;
        float s = __ldg(&scale[c]);
        int o = std::round(__ldg(&offset[c]));

        auto qt = QuantizeScalar<float, float, int>(
            __ldg(&value[index_offset]), s, o, clip_min, clip_max, rounding);
        float deq = DequantizeScalar<int, float, int>(qt, s, o);
        out[index_offset] = deq;
    }
}

template<int VPT, int TPB>
__global__ void _QuantizeTensorVectorize_LC(
    const int32_t    num_of_elements,
    const int32_t    element_per_channel,
    const int32_t    num_of_channel,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float*           out
){
    const int32_t index_offset = blockIdx.x * TPB * VPT + threadIdx.x * VPT;
    if (index_offset < num_of_elements){
        int c = (index_offset / element_per_channel) % num_of_channel;
        float s = __ldg(&scale[c]);
        int o = std::round(__ldg(&offset[c]));

        // never do (1 / s) here
        // value / s not always == value * (1 / s)
        # pragma unroll
        for(int32_t i = 0; i < VPT; i++){
            float qt = __ldg(&value[index_offset + i]) / s;
            int dq = CLIP<int>(_round2int(qt, rounding) + o, clip_min, clip_max);
            out[index_offset + i] = (dq - o) * s;
        }
    }
}

__host__ Tensor QuantizeTensor_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max, const int channel_axis,
    const Rounding rounding){
    /**
     * PPQ Tensor Quantization Function implementation.
     * This function quantizes a float tensor(channel wise),
     * giving an quantized tensor as output
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = (clip(i / s + o) - o) * s
     * Where s is scale factor, and o is offset
     *
     * This function is channel wise quantization function
     * Each channel has its own scale and offset.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");
    auto v_contiguous = value.contiguous();
    if(value.numel() > 0x7fffffff) throw InvalidValueException("There are too many element in your tensor(more than 2*10^9)"); 

    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 4;

    int32_t element_per_channel = v_contiguous.stride(channel_axis);
    int32_t num_of_channel      = v_contiguous.sizes()[channel_axis];
    int32_t num_of_elements     = NUM_OF_ELEMENT(v_contiguous);
    Tensor quantized            = at::empty_like(v_contiguous);

    if(element_per_channel % VPT == 0){
        _QuantizeTensorVectorize_LC<VPT, TPB>
        <<<NUM_OF_BLOCK_NOLIMIT(num_of_elements, VPT * TPB), TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_of_elements, element_per_channel, num_of_channel, PTR<float>(v_contiguous), PTR<float>(scale), PTR<float>(offset),
            clip_min, clip_max, rounding, PTR<float>(quantized)
        );
    }
    else{
        _QuantizeTensor_LC<TPB>
        <<<NUM_OF_BLOCK_NOLIMIT(num_of_elements, TPB), TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_of_elements, element_per_channel, num_of_channel,
            PTR<float>(v_contiguous), PTR<float>(scale), PTR<float>(offset),
            clip_min, clip_max, rounding, PTR<float>(quantized));
    }
    return quantized;
}

template<int VPT>
__global__
void _QuantizeTensor_LT_B(
    const int num_of_elements,
    const float* value,
    const float* scale,
    const float* offset,
    const float* grad_y,
    const int clip_min,
    const int clip_max,
    const float grad_factor,
    const Rounding rounding,
    float* grad_s,
    float* grad_x
){
    int32_t index = blockIdx.x * VPT + threadIdx.x; 

    if (index < num_of_elements){
        float o = std::round(__ldg(&offset[0]));
        float s = __ldg(&scale[0]);

        float v  = __ldg(&value[index]);
        float dy = __ldg(&grad_y[index]);

        int qt = _round2int(__ldg(&value[index]) / s, rounding) + o;

        // Calculate grad for scale and value
        float partial_gard_s = 0.0f;
        if(qt > clip_max) {
            partial_gard_s = (clip_max - o) * dy;
            grad_x[index] = 0;
        }
        else if(qt < clip_min) {
            partial_gard_s = (clip_min - o) * dy;
            grad_x[index] = 0;
        } else {
            auto q = DequantizeScalar<int, float, int>(qt, s, o);
            partial_gard_s = (q - v) * dy / s;
            grad_x[index] = dy;
        }

        // Reduce Gradient
        float reduced_grad_s = BlockReduceSum<float>(partial_gard_s);
        if (threadIdx.x == 0) {
            atomicAdd(grad_s, reduced_grad_s * grad_factor);
        }
    }
}

__host__ std::vector<Tensor> QuantizeTensor_LT_B(
    const Tensor &value, const Tensor &scale, const Tensor &offset, const Tensor &grad_y,
    const int clip_min, const int clip_max, const Rounding rounding
){
    //
    // Gradient Bakcwrad for quantization
    // Solve grad_s, grad_v in one function.
    //
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");
    auto v_contiguous = value.contiguous();
    auto grad_contiguous = grad_y.contiguous();

    int32_t num_of_element = NUM_OF_ELEMENT(v_contiguous);
    if(num_of_element > 0x7fffffff) 
        throw InvalidValueException("There are too many element in your tensor(more than 2*10^9)");
    
    constexpr int TPB = 1024;
    Tensor grad_s = at::zeros_like(scale);
    Tensor grad_x = at::empty_like(grad_contiguous);
    float grad_factor = rsqrtf(((double)num_of_element * (clip_max - clip_min)));

    _QuantizeTensor_LT_B<TPB>
        <<<NUM_OF_BLOCK_NOLIMIT(num_of_element, TPB), TPB, 0, at::cuda::getCurrentCUDAStream()>>>
    (
        NUM_OF_ELEMENT(v_contiguous),
        PTR<float>(v_contiguous),
        PTR<float>(scale),
        PTR<float>(offset),
        PTR<float>(grad_contiguous),
        clip_min,
        clip_max,
        grad_factor,
        rounding,
        PTR<float>(grad_s),
        PTR<float>(grad_x)
    );
    return {grad_x, grad_s};
}

template<int VPT>
__global__
void _QuantizeTensor_LC_B(
    const int32_t num_of_elements,
    const int32_t element_per_channel,
    const int32_t num_of_channel,
    const float*  value,
    const float*  scale,
    const float*  offset,
    float*        grad_y,
    const int     clip_min,
    const int     clip_max,
    const float   grad_factor,
    const Rounding rounding,
    float *       grad_s,
    float *       grad_x
){
    int32_t channel_idx = blockIdx.x;
    int32_t iter = (channel_idx * element_per_channel) + blockIdx.y * VPT + threadIdx.x;
    float s = scale[channel_idx];
    float o = std::round(offset[channel_idx]);

    float partial_gard_s = 0;
    // if processing element is not belongs to correct channel, skip its computation
    while ((blockIdx.y * VPT + threadIdx.x < element_per_channel) && iter < num_of_elements){

        float dy = __ldg(&grad_y[iter]);
        float v  = __ldg(&value[iter]);

        grad_x[iter] = dy;

        int qt = _round2int(__ldg(&value[iter]) / s, rounding) + o;
        // Calculate grad for scale and value
        if(qt > clip_max) {
            partial_gard_s += (clip_max - o) * dy;
            grad_x[iter] = 0;
        }
        else if(qt < clip_min) {
            partial_gard_s += (clip_min - o) * dy;
            grad_x[iter] = 0;
        } else {
            auto q = DequantizeScalar<int, float, int>(qt, s, o);
            partial_gard_s += (q - v) / s * dy;
            grad_x[iter] = dy;
        }

        iter += element_per_channel * num_of_channel;
    }

    // Reduce Gradient
    float reduced_grad_s = BlockReduceSum<float>(partial_gard_s);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_s[channel_idx], reduced_grad_s * grad_factor);
    }
}


__host__ std::vector<Tensor> QuantizeTensor_LC_B(
    const Tensor &value, const Tensor &scale, const Tensor &offset, 
    const Tensor &grad_y, const int clip_min, const int clip_max,
    const Rounding rounding, const int channel_axis
){
    /**
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");

    auto v_contiguous = value.contiguous();
    auto grad_contiguous = grad_y.contiguous();
    int32_t num_of_element = NUM_OF_ELEMENT(value);

    if(num_of_element > 0x7fffffff) throw InvalidValueException("There are too many element in your tensor(more than 2*10^9)");
    float grad_factor = rsqrtf(((double)num_of_element * clip_max));

    Tensor grad_s = at::zeros_like(scale);
    Tensor grad_x = at::empty_like(grad_y);

    int32_t element_per_channel = v_contiguous.stride(channel_axis);
    int32_t num_of_channel      = v_contiguous.sizes()[channel_axis];
    constexpr int32_t VPT       = 1024;

    dim3 grid;
    grid.x = static_cast<uint32_t>(num_of_channel);
    grid.y = static_cast<uint32_t>(NUM_OF_BLOCK_NOLIMIT(element_per_channel, VPT));
    grid.z = 1;

    _QuantizeTensor_LC_B<VPT>
        <<<grid, VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_of_element,
        element_per_channel,
        num_of_channel,
        PTR<float>(v_contiguous),
        PTR<float>(scale),
        PTR<float>(offset),
        PTR<float>(grad_y),
        clip_min,
        clip_max,
        grad_factor,
        rounding,
        PTR<float>(grad_s),
        PTR<float>(grad_x)
    );
    return {grad_x, grad_s};
}
