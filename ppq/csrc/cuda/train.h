# include "common.cuh"

 Tensor TensorClip_T(
    const Tensor &value, const Tensor &reference, const Tensor &limit);

 Tensor TensorClip_C(
    const Tensor &value, const Tensor &reference, const Tensor &limit,
    const int channel_axis);

Tensor RoundingLoss_LT(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    Rounding rounding);

Tensor RoundingLoss_LT_B(
    const Tensor &value, const Tensor &dy,
    const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    Rounding rounding);

Tensor RoundingLoss_LC(
    const Tensor &value, const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    const int channel_axis, Rounding rounding);

Tensor RoundingLoss_LC_B(
    const Tensor &value, const Tensor &dy,
    const Tensor &scale, const Tensor &offset,
    const int clip_min, const int clip_max,
    const int channel_axis, Rounding rounding);
