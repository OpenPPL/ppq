# include "common.cuh"

Tensor QuantizeTensor_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max, const Rounding rounding);

Tensor QuantizeTensor_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max, const int channel_axis,
    const Rounding rounding);

std::vector<Tensor> QuantizeTensor_LT_B(
    const Tensor &value, const Tensor &scale, 
    const Tensor &offset, const Tensor &grad_y,
    const int clip_min, const int clip_max, 
    const Rounding rounding);

std::vector<Tensor> QuantizeTensor_LC_B(
    const Tensor &value, const Tensor &scale, 
    const Tensor &offset, const Tensor &grad_y,
    const int clip_min, const int clip_max,
    const Rounding rounding, const int channel_axis);
