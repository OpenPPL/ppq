# include "train.h"

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

__global__ void _TensorClip_C(
    const int64_t    num_of_element,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float*     value,
    const float*     reference,
    const float*     limit,
    float *          out
){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        int c = (iter / element_per_channel) % num_of_channel;
        out[iter] = CLIP<float>(value[iter], reference[iter] - limit[c], reference[iter] + limit[c]);
    }
}

__host__ Tensor TensorClip_C(
    const Tensor &value, const Tensor &reference, const Tensor &limit,
    const int channel_axis){
    /**
    Clip a tensor inplace with given limit
    All scalar of value will be cliped with range [reference - limit, reference + limit]

    This function will clip tensor value inplace.
     */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(reference, at::kFloat, "Reference(Expect to be FP32)");
    CheckTensor(limit, at::kFloat, "Limit(Expect to be FP32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    Tensor out = at::empty_like(value);
    _TensorClip_C<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        PTR<float>(value), PTR<float>(reference), PTR<float>(limit),
        PTR<float>(out)
    );
    return out;
}

__global__ void _TensorClip_T(
    const int64_t    num_of_element,
    const float*     value,
    const float*     reference,
    const float*     limit,
    float *          out
){
    int64_t iter; float l = limit[0];
    KERNEL_LOOP(iter, num_of_element){
        out[iter] = CLIP<float>(value[iter], reference[iter] - l, reference[iter] + l);
    }
}

__host__ Tensor TensorClip_T(
    const Tensor &value, const Tensor &reference, const Tensor &limit){
    /*
    Clip a tensor inplace with given limit
    All scalar of value will be cliped with range [reference - limit, reference + limit]

    This function will clip tensor value inplace.
    */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(reference, at::kFloat, "Reference(Expect to be FP32)");
    CheckTensor(limit, at::kFloat, "Limit(Expect to be FP32)");

    Tensor out = at::empty_like(value);
    _TensorClip_T<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        PTR<float>(value), PTR<float>(reference), PTR<float>(limit),
        PTR<float>(out)
    );
    return out;
}

__global__ void _RoundingLoss_LT(
    const int64_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* out
){
    float s = scale[0]; int o = std::nearbyint(offset[0]); int64_t iter;
    float diff_sum = 0;
    KERNEL_LOOP(iter, num_of_element){
        float v = value[iter];
        auto _ = QuantizeScalar<float, float, int>(
            v, s, o, clip_min, clip_max, rounding);
        auto out = DequantizeScalar<int, float, int>(_, s, o);

        float diff = abs(out - v);
        // if value has been clipped, set diff = 0
        if (v > s * (clip_max - o)) diff = 0;
        if (v < s * (clip_min - o)) diff = 0;
        diff_sum += diff;
    }
    float diff_reduced = BlockReduceSum<float>(diff_sum);
    if(threadIdx.x == 0) atomicAdd(out, diff_reduced / sqrtf(num_of_element));
}

__host__ Tensor RoundingLoss_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    Rounding rounding
){
    /*
        Compute rounding loss R of a given tensor T.

        R = sum(norm2(T - quant(T))) / sqrt(elements_of(T))

        If one value inside R is clipped by quant function,
            then it contributes nothing to R.
    */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    Tensor loss = at::zeros({1}, value.options());
    _RoundingLoss_LT<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, PTR<float>(loss)
    );
    return loss;
}

__global__ void _RoundingLoss_LT_B(
    const int64_t    num_of_element,
    const float*     value,
    const float*     dy,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* dx
){
    float s = scale[0]; int o = std::nearbyint(offset[0]); int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        float v = value[iter];
        auto _ = QuantizeScalar<float, float, int>(
            v, s, o, clip_min, clip_max, rounding);
        auto out = DequantizeScalar<int, float, int>(_, s, o);

        float grad = ((v > out) ? 1 : -1) * dy[0];
        // if value has been clipped, set grad = 0
        if (v > s * (clip_max - o)) grad = 0;
        if (v < s * (clip_min - o)) grad = 0;
        dx[iter] = grad / sqrtf(num_of_element);
    }
}

__host__ Tensor RoundingLoss_LT_B(
    const Tensor &value, const Tensor &dy,
    const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    Rounding rounding
){
    /*
        Compute grad through rounding loss.
    */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    Tensor grad_v = at::zeros_like(value);
    _RoundingLoss_LT_B<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value),
        PTR<float>(value), PTR<float>(dy),
        PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, PTR<float>(grad_v)
    );
    return grad_v;
}

__global__ void _RoundingLoss_LC(
    const int64_t    num_of_element,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* out
){
    float diff_sum = 0; int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        float v = value[iter]; int c = (iter / element_per_channel) % num_of_channel;
        auto _ = QuantizeScalar<float, float, int>(
            value[iter], scale[c], std::nearbyint(offset[c]), clip_min, clip_max, rounding);
        auto out = DequantizeScalar<int, float, int>(_, scale[c], std::nearbyint(offset[c]));

        float diff = abs(out - v);
        // if value has been clipped, set diff = 0
        if (v > scale[c] * (clip_max - offset[c])) diff = 0;
        if (v < scale[c] * (clip_min - offset[c])) diff = 0;
        diff_sum += diff;
    }
    float diff_reduced = BlockReduceSum<float>(diff_sum);
    if(threadIdx.x == 0) atomicAdd(out, diff_reduced / sqrtf(num_of_element));
}

__host__ Tensor RoundingLoss_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    const int channel_axis, Rounding rounding
){
    /*
        Compute rounding loss R of a given tensor T.

        R = sum(norm2(T - quant(T))) / sqrtf(elements_of(T))

        If one value inside R is clipped by quant function,
            then it contributes nothing to R.
    */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    Tensor loss = at::zeros({1}, value.options());
    _RoundingLoss_LC<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        PTR<float>(value), PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, PTR<float>(loss)
    );
    return loss;
}

__global__ void _RoundingLoss_LC_B(
    const int64_t    num_of_element,
    const int64_t    element_per_channel,
    const int        num_of_channel,
    const float*     value,
    const float*     dy,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float* dx
){
    int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        float v = value[iter]; int c = (iter / element_per_channel) % num_of_channel;
        auto _ = QuantizeScalar<float, float, int>(
            value[iter], scale[c], std::nearbyint(offset[c]), clip_min, clip_max, rounding);
        auto out = DequantizeScalar<int, float, int>(_, scale[c], std::nearbyint(offset[c]));

        float grad = ((v > out) ? 1 : -1) * dy[0];
        // if value has been clipped, set grad = 0
        if (v > scale[c] * (clip_max - offset[c])) grad = 0;
        if (v < scale[c] * (clip_min - offset[c])) grad = 0;
        dx[iter] = grad / sqrtf(num_of_element);
    }
}

__host__ Tensor RoundingLoss_LC_B(
    const Tensor &value, const Tensor &dy,
    const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    const int channel_axis, Rounding rounding
){
    /*
        Compute grad through rounding loss.
    */
    CheckTensor(value, at::kFloat, "Value(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(offset, at::kFloat, "Offset(Expect to be FP32)");

    int element_per_channel = 1;
    const int num_of_channel = value.sizes()[channel_axis];
    for(int axis = value.ndimension() - 1; axis != channel_axis; axis--){
        element_per_channel *= value.sizes()[axis];
    }

    Tensor grad_v = at::zeros_like(value);
    _RoundingLoss_LC_B<<<NUM_OF_BLOCK(NUM_OF_ELEMENT(value), CUDA_NUM_THREADS),
        CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        NUM_OF_ELEMENT(value), element_per_channel, num_of_channel,
        PTR<float>(value), PTR<float>(dy),
        PTR<float>(scale), PTR<float>(offset),
        clip_min, clip_max, rounding, PTR<float>(grad_v)
    );
    return grad_v;
}
