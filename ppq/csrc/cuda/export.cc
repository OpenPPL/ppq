# include "linear.h"
# include "sort.h"
# include "train.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Quantile_T", Quantile_T, "Quantile_T");
    m.def("Histogram_T", Histogram_T, "Histogram_T");
    m.def("Histogram_C", Histogram_C, "Histogram_C");
    
    m.def("QuantizeTensor_LT", QuantizeTensor_LT, "QuantizeTensor_LT");
    m.def("QuantizeTensor_LC", QuantizeTensor_LC, "QuantizeTensor_LC");
    m.def("QuantizeTensor_LT_B", QuantizeTensor_LT_B, "QuantizeTensor_LT_B");
    m.def("QuantizeTensor_LC_B", QuantizeTensor_LC_B, "QuantizeTensor_LC_B");

    m.def("TensorClip_T", TensorClip_T, "TensorClip_T");
    m.def("TensorClip_C", TensorClip_C, "TensorClip_C");

    m.def("RoundingLoss_LT", RoundingLoss_LT, "RoundingLoss_LT");
    m.def("RoundingLoss_LC", RoundingLoss_LC, "RoundingLoss_LC");
    m.def("RoundingLoss_LT_B", RoundingLoss_LT_B, "RoundingLoss_LT_B");
    m.def("RoundingLoss_LC_B", RoundingLoss_LC_B, "RoundingLoss_LC_B");
}