# include "common.cuh"

Tensor QuantizeTensor_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset, 
    const int clip_min, const int clip_max, 
    Rounding rounding);

Tensor QuantizeTensor_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset, 
    const int clip_min, const int clip_max, const int channel_axis,
    Rounding rounding);

void Histogram_T(
    const Tensor &value,
    const float hist_scale,
    const bool clip_outliers,
    Tensor &hist);

std::vector<Tensor> QuantizeTensor_LT_B(
    const Tensor &value, const Tensor &quantized, 
    const Tensor &scales, const Tensor &offsets, const Tensor &grad_y, 
    const int clip_min, const int clip_max);

std::vector<Tensor> QuantizeTensor_LC_B(
    const Tensor &value, const Tensor &quantized, 
    const Tensor &scales, const Tensor &offsets, const Tensor &grad_y, 
    const int clip_min, const int clip_max, const int channel_axis);