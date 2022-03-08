# include <cuda.h>
# include <cuda_runtime.h>
# include <math.h>
# include <torch/extension.h>
# include <ATen/cuda/CUDAContext.h>
# pragma once

using at::Tensor;
using Rounding = int;

# define __inline__ inline
constexpr int64_t CUDA_NUM_THREADS     = 512;
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
int64_t NUM_OF_BLOCK(int64_t elements){
    return std::min((elements + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_TARGET_BLOCKS);
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
        throw ValueTypeException(
            std::move("Kernel Failure, Invalid dtype of Input tensor: " + name));
    }
    if(tensor.numel() == 0){
        throw InvalidValueException(
            std::move("Kernel Failure, Tensor is empty: " + name));
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

