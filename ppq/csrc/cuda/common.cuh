# include <cuda.h>
# include <cuda_runtime.h>
# include <math.h>
# include <torch/extension.h>
# include <ATen/cuda/CUDAContext.h>
# pragma once

using at::Tensor;
using Rounding = int;
using std::max;
using std::min;

# define __inline__ inline
constexpr int64_t CUDA_NUM_THREADS     = 1024;
constexpr int64_t CUDA_TARGET_BLOCKS   = 2560;

constexpr int ROUND_HALF_EVEN          = 0;
constexpr int ROUND_HALF_UP            = 1;
constexpr int ROUND_HALF_DOWN          = 2;
constexpr int ROUND_HALF_TOWARDS_ZERO  = 3;
constexpr int ROUND_HALF_FAR_FORM_ZERO = 4;
constexpr int ROUND_TO_NEAR_INT        = 5;
constexpr int ROUND_UP                 = 6;
constexpr int ROUND_DOWN               = 7;

constexpr unsigned int FINAL_MASK      = 0xffffffff;

# define KERNEL_LOOP(i, j) for(i = blockIdx.x * blockDim.x + threadIdx.x; i < j; i += blockDim.x * gridDim.x)
# define BLOCKIDX_3D ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z)
# define BLOCKIDX_2D (blockIdx.x * gridDim.y + blockIdx.y)

class ValueTypeException: public std::exception {
public:
    explicit ValueTypeException(const char *m) : message{m} {}
    explicit ValueTypeException(const std::string m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }
private:
    std::string message = "";
};

class InvalidValueException: public std::exception {
public:
    explicit InvalidValueException(const char *m) : message{m} {}
    explicit InvalidValueException(const std::string m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }
private:
    std::string message = "";
};

template<typename Dtype>
__host__ __inline__ Dtype* PTR(Tensor t){
    return t.data_ptr<Dtype>();
}

__host__ __inline__
int64_t NUM_OF_ELEMENT(Tensor t){
    return t.numel();
}

__host__ __inline__
int64_t NUM_OF_BLOCK(const int64_t elements, const int threads_per_block){
    return std::min((elements + threads_per_block - 1) / threads_per_block, CUDA_TARGET_BLOCKS);
}

__host__ __inline__
int64_t NUM_OF_BLOCK_NOLIMIT(const int64_t elements, const int threads_per_block){
    return (elements + threads_per_block - 1) / threads_per_block;
}

template<typename Dtype>
__device__ __inline__
Dtype CLIP(const Dtype v, const Dtype min, const Dtype max){
    if(v > max) return max;
    if(v < min) return min;
    return v;
}

__host__ __inline__
void CheckTensor(const Tensor &tensor, const c10::ScalarType &type, const std::string &name){
    if(at::typeMetaToScalarType(tensor.dtype()) != type){
        throw ValueTypeException("Kernel Failure, Invalid dtype of Input tensor: " + name);
    }
    if(tensor.numel() == 0){
        throw InvalidValueException("Kernel Failure, Tensor is empty: " + name);
    }
}

__device__ __inline__
int _round2int(
    const float value,
    const int rounding
){
    switch(rounding){
        case ROUND_HALF_EVEN:
            return std::nearbyint(value);
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
            return round(value);
    }
    return 0;
}

template<typename Dtype, typename Stype, typename Otype>
__device__ __inline__
int QuantizeScalar(
    const Dtype value, const Stype scale, const Otype offset,
    const int clip_min, const int clip_max,
    const Rounding rounding){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an float value to int32
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = clip(i / s + o)
     * Where s is scale factor, and o is offset
     */
    int v = _round2int(value / scale, rounding) + offset;
    return CLIP<int>(v, clip_min, clip_max);
}

template<typename Dtype, typename Stype, typename Otype>
__device__ __inline__
float DequantizeScalar(
    const Dtype value, const Stype scale, const Otype offset){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an int32 value to float
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = (i - o) * s
     * Where s is scale factor, and o is offset
     */
    return (value - offset) * scale;
}

union FPConvertHelper {
    float value;
    uint32_t data;
};

template<typename Dtype, typename Stype, typename Otype>
__device__ __inline__
float QuantizeScalarFloating(
    const Dtype value, const Stype scale, const Otype offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, 
    const Rounding rounding){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an float value to low-precision float
     */
    FPConvertHelper helper; FPConvertHelper rounding_helper;
    float Unscaled_FP32 = static_cast<float>(value) / scale;
    
    helper.value = Unscaled_FP32;
	int32_t exponent_min  = -(1 << (exponent - 1)) + 1;
    int32_t exponent_max  = (1 << (exponent - 1));

    // Following code will process exponent overflow
    /* For FP8 E4M3, the maximum exponent value should be 8.                                  */
    /* The Maximum number of FP8 E4M3 should be (0 1111 111) = 480                            */
    /* We call it as theoretical_maximum, FP8 E4M3 can not represent a number larger than it. */
    uint32_t fp32_sign    = 0;
    int32_t fp32_exp      = (exponent_max + 127) << 23;
    int32_t fp32_mantissa = ~(0x007FFFFF >> mantissa) & 0x007FFFFF;
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;
    float theoretical_maximum = helper.value;

    if (Unscaled_FP32 > min(clip_max, theoretical_maximum)) 
        return min(clip_max, theoretical_maximum);
    if (Unscaled_FP32 < max(clip_min, -theoretical_maximum)) 
        return max(clip_min, -theoretical_maximum);

    // Code start from here will convert number within fp8 range.
    // Following code will Split float32 into sign, exp, mantissa
    /* IEEE 754 Standard: 1 bit sign, 8 bit exponent, 23 bit mantissa */

    /* In binary 10000000 00000000 00000000 00000000 = 0x80000000 in Hex */
    /* In binary 01111111 10000000 00000000 00000000 = 0x7F800000 in Hex */
    /* In binary 00000000 01111111 11111111 11111111 = 0x007FFFFF in Hex */

    /* Tool: https://www.h-schmidt.net/FloatConverter/IEEE754.html */
    helper.value  = Unscaled_FP32;
    fp32_sign     = helper.data & 0x80000000;
    fp32_exp      = helper.data & 0x7F800000;
    fp32_mantissa = helper.data & 0x007FFFFF;

    // Following code will process exponent underflow
    /* Float underflow means fp32_exp is smaller than exponent_min          */
    /* Where exponent_min is the minimum exponent value of quantized float. */
    /* For FP8 E4M3, the minimum exponent value should be -7.               */
    /* The Min Subnormal value of FP8 E4M3 should be (0 0000 001) = 2^-9    */
    /* The Min normal value of FP8 E4M3 should be (0 0001 000) = 2^-6       */
	if (((fp32_exp >> 23) - 127) < exponent_min + 1){
        // following divide might have some problems
        // but it is the simplest method with very limited error.
        float min_subnormal = 1.0f / (1 << ((1 << (exponent - 1)) + mantissa - 2));
        return _round2int(Unscaled_FP32 / min_subnormal, rounding) * min_subnormal;
	}

    /* high precision mantissa convert to low precision mantissa requires rounding                         */
    /* Here we apply a tricky method to round mantissa:                                                    */
    /* We create another float, which sign = 0, exponent = 127, mantissa = fp32_mantissa << (23 - mantissa) */
    /* Then we directly round this float to int, result here is what we want, you can prove it by yourself */
    rounding_helper.data = ((fp32_mantissa << (mantissa)) & 0x007FFFFF) + 0x3F800000;
    uint32_t round_bit = _round2int(rounding_helper.value - 1, rounding);

    // process mantissa
    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa)) + round_bit) << (23 - mantissa);
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;

    return CLIP<float>(helper.value, clip_min, clip_max);
}