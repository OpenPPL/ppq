# include "common.h"

void cuda_linear_tensor_quantize(
    const Tensor &source,
    Tensor &dest,
    const float scale,
    const float offset,
    const int minimum,
    const int maximum,
    const int rounding
);

void cuda_linear_channel_quantize(
    const Tensor &source,
    const Tensor &scales,
    const Tensor &offsets,
    Tensor &dest,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding
);

void cuda_tensor_histogram(
    const Tensor &source,
    Tensor &dest,
    const float scale,
    const int offset,
    const bool abs,
    const int rounding
);