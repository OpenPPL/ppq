# include "common.h"
# include <vector>
using std::vector;

vector<Tensor> cuda_linear_tensor_quant_sieve(
    const Tensor &tensor,
    const Tensor &fp_offset_tensor,
    const float scale,
    const int offset,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
);

vector<Tensor> cuda_linear_channel_quant_sieve(
    const Tensor &tensor,
    const Tensor &fp_offset_tensor,
    const Tensor &scales,
    const Tensor &offsets,
    const int channel_axis,
    const int minimum,
    const int maximum,
    const int rounding,
    const float offset_limit,
    const float threshold
);

