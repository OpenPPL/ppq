# include "common.cuh"

Tensor cuda_linear_tensor_gaussian_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    const int minimum,
    const int maximum,
    const int rounding,
    const float sigma
);

Tensor cuda_linear_channel_gaussian_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding,
    const float sigma
);