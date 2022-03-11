# include "common.cuh"

__device__
inline float MeanSquareError(const float a, const float b){
    return (a - b) * (a - b);
}

__global__
void _QuantizeTensor_LT(
    const int64_t    num_of_element,
    const float*     value,
    const float*     scale,
    const float*     offset,
    const int        clip_min,
    const int        clip_max,
    const Rounding   rounding,
    float*           out
){
    int64_t threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    float s_multipiler = 0; int offset_bias = 0;
    float s = scale[0]; int o = std::nearbyint(offset[0]); int64_t iter;
    KERNEL_LOOP(iter, num_of_element){
        auto _ = QuantizeScalar<float, float, int>(
            value[iter], s, o, clip_min, clip_max, rounding);
        out[iter] = DequantizeScalar<int, float, int>(_, s, o);
    }
}