# include "common.cuh"

Tensor Quantile_T(const Tensor &source, const float q);

void Histogram_T(
    const Tensor &value,
    const float hist_scale,
    const bool clip_outliers,
    Tensor &hist);

void Histogram_C(
    const Tensor &value,
    const int channel_axis,
    const float hist_scale,
    const bool clip_outliers,
    Tensor &hist);