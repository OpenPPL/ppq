# include "linear.h"
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

__global__ void _QuantizeTensor_LT(
    const int64_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    const float      dropout,
    float* out
){
    float s = scale[0]; int o = std::nearbyint(offset[0]); int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        float qt = QuantizeScalar<float, float, int>(
            value[iter], s, o, clip_min, clip_max, rounding);
        float deq = DequantizeScalar<int, float, int>(qt, s, o);

        // Qdrop, Notice here out[iter] is initilized with torch.rand
        bool mask = dropout > 0 && dropout > out[iter];
        out[iter] = value[iter] * mask + deq * (1 - mask);
    }
}

__host__ Tensor QuantizeTensor_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset, 
    const int clip_min, const int clip_max, const Rounding rounding,  
    const float dropout){
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
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(scale, at::kFloat, "Scale(FP32)");
    CheckTensor(offset, at::kFloat, "Offset(FP32)");

    Tensor quantized;
    if (dropout == 0) quantized = at::empty_like(value);
    else quantized = at::rand_like(value);

    _QuantizeTensor_LT<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value)), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, dropout, PTR<float>(quantized)
    );
    return quantized;
}

__global__ void _QuantizeTensor_LC(
    const int64_t    num_of_element,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    const float      dropout,
    float* out
){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        int c = (iter / element_per_channel) % num_of_channel;
        auto qt = QuantizeScalar<float, float, int>(
            value[iter], scale[c], std::nearbyint(offset[c]), clip_min, clip_max, rounding);
        float deq = DequantizeScalar<int, float, int>(qt, scale[c], std::nearbyint(offset[c]));

        // Qdrop, Notice here out[iter] is initilized with torch.rand
        bool mask = dropout > 0 && dropout > out[iter];
        out[iter] = value[iter] * mask + deq * (1 - mask);
    }
}

__host__ Tensor QuantizeTensor_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset, 
    const int clip_min, const int clip_max, const int channel_axis,
    const Rounding rounding, const float dropout){
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
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(scale, at::kFloat, "Scale(FP32)");
    CheckTensor(offset, at::kFloat, "Offset(FP32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    Tensor quantized;
    if (dropout == 0) quantized = at::empty_like(value); // torch empty like will not invoke actual kernel.
    else quantized = at::rand_like(value);

    _QuantizeTensor_LC<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value)), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, dropout, PTR<float>(quantized)
    );
    return quantized;
}

__global__
void _QuantizeTensor_LT_B(
    const int64_t num_of_elements,
    const float* value,
    const float* quantized,
    const float* scales,
    const float* offsets,
    const float* grad_y,
    const int clip_min,
    const int clip_max,
    float * grad_v,
    float * grad_s,
    float * grad_o
){
    int64_t iter; float s = scales[0]; int o = std::nearbyint(offsets[0]);

    KERNEL_LOOP(iter, num_of_elements){
        float v = value[iter];
        
        /* Calculate grad for scales, offsets, alpha */
        float partial_gard_s = ((quantized[iter] - v) / s) * grad_y[iter];
        float partial_gard_o = 0.0f;
        float partial_gard_v = grad_y[iter];;

        if(v > s * (clip_max - o)) {
            partial_gard_s = 1 * grad_y[iter];
            partial_gard_o = -grad_y[iter] * s;
            partial_gard_v = 0;
        }
        else if(v < s * (clip_min - o)) {
            partial_gard_s = -1 * grad_y[iter];
            partial_gard_o = -grad_y[iter] * s;
            partial_gard_v = 0;
        }
        grad_v[iter] = partial_gard_v;
        
        /* Reduce Gradient */
        float reduced_grad_o = BlockReduceSum<float>(partial_gard_o); __syncthreads();
        float reduced_grad_s = BlockReduceSum<float>(partial_gard_s); __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(grad_s, reduced_grad_s);
            atomicAdd(grad_o, reduced_grad_o);
        }
    }
}

__host__ std::vector<Tensor> QuantizeTensor_LT_B(
    const Tensor &value, const Tensor &quantized, 
    const Tensor &scales, const Tensor &offsets, const Tensor &grad_y, 
    const int clip_min, const int clip_max
){
    /** 
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(quantized, at::kFloat, "Quantized(FP32)");
    CheckTensor(scales, at::kFloat, "Scale(FP32)");
    CheckTensor(offsets, at::kFloat, "Offset(FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(FP32)");

    Tensor grad_v = at::zeros_like(value);
    Tensor grad_s = at::zeros_like(scales);
    Tensor grad_o = at::zeros_like(scales);

    _QuantizeTensor_LT_B<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value)), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        PTR<float>(value),
        PTR<float>(quantized),
        PTR<float>(scales),
        PTR<float>(offsets),
        PTR<float>(grad_y),
        clip_min,
        clip_max,
        PTR<float>(grad_v),
        PTR<float>(grad_s),
        PTR<float>(grad_o)
    );
    return {grad_v, grad_s, grad_o};
}

__global__
void _QuantizeTensor_LC_B(
    const int64_t    num_of_elements,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float* value,
    const float* quantized,
    const float* scales,
    const float* offsets,
    const float* grad_y,
    const int clip_min,
    const int clip_max,
    float * grad_v,
    float * grad_s,
    float * grad_o
){
    int64_t iter; 
    int64_t iter_offset = (blockIdx.x * element_per_channel * num_of_channel) + 
                          (blockIdx.y * element_per_channel);
    float partial_gard_o = 0; float partial_gard_s = 0;
    int c = blockIdx.y; float s = scales[c]; int o = std::nearbyint(offsets[c]);

    for(iter = (blockIdx.x * blockDim.x) + threadIdx.x; 
        // if processing element is not belongs to correct channel, skip its computation
        iter < element_per_channel; 
        iter += blockDim.x * gridDim.x)
    {
        float g = grad_y[iter + iter_offset];
        
        /* Calculate grad for scales, offsets, alpha */
        partial_gard_s = (quantized[iter + iter_offset] - value[iter + iter_offset]) * (g / s);
        float partial_gard_v = g;

        if(value[iter + iter_offset] > s * (clip_max - o)) {
            partial_gard_s = 1 * g;
            partial_gard_o = -g * s;
            partial_gard_v = 0;
        }
        else if(value[iter + iter_offset] < s * (clip_min - o)) {
            partial_gard_s = -1 * g;
            partial_gard_o = -g * s;
            partial_gard_v = 0;
        }

        grad_v[iter + iter_offset] = partial_gard_v;

        /* Reduce Gradient */
        float reduced_grad_o = BlockReduceSum<float>(partial_gard_o); __syncthreads();
        float reduced_grad_s = BlockReduceSum<float>(partial_gard_s); __syncthreads();

        if (threadIdx.x == 0) {
            // printf("ro=%.2f\t rs=%.2f\t c=%d\t\n", reduced_grad_o, reduced_grad_s, c);
            atomicAdd(&grad_o[c], reduced_grad_o);
            atomicAdd(&grad_s[c], reduced_grad_s);
        }
    }
}

__host__ std::vector<Tensor> QuantizeTensor_LC_B(
    const Tensor &value, const Tensor &quantized, 
    const Tensor &scales, const Tensor &offsets, const Tensor &grad_y, 
    const int clip_min, const int clip_max, const int channel_axis
){
    /** 
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(value, at::kFloat, "Value(FP32)");
    CheckTensor(quantized, at::kFloat, "Quantized(FP32)");
    CheckTensor(scales, at::kFloat, "Scale(FP32)");
    CheckTensor(offsets, at::kFloat, "Offset(FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(FP32)");

    Tensor grad_v = at::zeros_like(value);
    Tensor grad_s = at::zeros_like(scales);
    Tensor grad_o = at::zeros_like(scales);

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    const int batchs = NUM_OF_ELEMENT(value) / (element_per_channel * num_of_channel);
    const dim3 grid_size(batchs, num_of_channel, NUM_OF_BLOCK(element_per_channel));
    _QuantizeTensor_LC_B<<<grid_size, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        element_per_channel,
        num_of_channel,
        PTR<float>(value),
        PTR<float>(quantized),
        PTR<float>(scales),
        PTR<float>(offsets),
        PTR<float>(grad_y),
        clip_min,
        clip_max,
        PTR<float>(grad_v),
        PTR<float>(grad_s),
        PTR<float>(grad_o)
    );
    return {grad_v, grad_s, grad_o};
}
