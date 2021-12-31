# include "linear.h"
# include "sort.h"
# include "sieve.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TensorwiseLinearQuantize", cuda_linear_tensor_quantize, "TensorwiseLinearQuantize");
    m.def("ChannelwiseLinearQuantize", cuda_linear_channel_quantize, "ChannelwiseLinearQuantize");
    
    m.def("TensorwiseHistogram", cuda_tensor_histogram, "TensorwiseHistogram");
    m.def("Quantile", cuda_quantile, "Quantile");

    m.def("TensorwiseLinearQuantSieve", cuda_linear_tensor_quant_sieve, "TensorwiseLinearQuantSieve");
    m.def("ChannelwiseLinearQuantSieve", cuda_linear_channel_quant_sieve, "ChannelwiseLinearQuantSieve");
}