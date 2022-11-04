# include "common.cuh"

Tensor QuantizeTensor_FT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, const Rounding rounding);

Tensor QuantizeTensor_FC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, const int channel_axis,
    const Rounding rounding);

std::vector<Tensor> QuantizeTensor_FT_B(
    const Tensor &value, const Tensor &scales, 
    const Tensor &offsets, const Tensor &grad_y,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, 
    const Rounding rounding);

std::vector<Tensor> QuantizeTensor_FC_B(
    const Tensor &value, const Tensor &scales, 
    const Tensor &offsets, const Tensor &grad_y,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max,
    const Rounding rounding, const int channel_axis);
