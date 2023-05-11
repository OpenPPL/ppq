# include "cuda/linear.h"
# include "cuda/sort.h"
# include "cuda/train.h"
# include "cuda/train.h"
# include "cuda/floating.h"
# include "cpu/hist_mse.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Quantile_T", Quantile_T, "Quantile_T");
    m.def("Histogram_T", Histogram_T, "Histogram_T");
    m.def("Histogram_Asymmetric_T", Histogram_Asymmetric_T, "Histogram_Asymmetric_T");
    m.def("Histogram_C", Histogram_C, "Histogram_C");

    m.def("QuantizeTensor_LT", QuantizeTensor_LT, "QuantizeTensor_LT");
    m.def("QuantizeTensor_LC", QuantizeTensor_LC, "QuantizeTensor_LC");
    m.def("QuantizeTensor_LT_B", QuantizeTensor_LT_B, "QuantizeTensor_LT_B");
    m.def("QuantizeTensor_LC_B", QuantizeTensor_LC_B, "QuantizeTensor_LC_B");

    m.def("QuantizeTensor_FT", QuantizeTensor_FT, "QuantizeTensor_FT");
    m.def("QuantizeTensor_FC", QuantizeTensor_FC, "QuantizeTensor_FC");
    m.def("QuantizeTensor_FT_B", QuantizeTensor_FT_B, "QuantizeTensor_FT_B");
    m.def("QuantizeTensor_FC_B", QuantizeTensor_FC_B, "QuantizeTensor_FC_B");

    m.def("TensorClip_T", TensorClip_T, "TensorClip_T");
    m.def("TensorClip_C", TensorClip_C, "TensorClip_C");

    m.def("RoundingLoss_LT", RoundingLoss_LT, "RoundingLoss_LT");
    m.def("RoundingLoss_LC", RoundingLoss_LC, "RoundingLoss_LC");
    m.def("RoundingLoss_LT_B", RoundingLoss_LT_B, "RoundingLoss_LT_B");
    m.def("RoundingLoss_LC_B", RoundingLoss_LC_B, "RoundingLoss_LC_B");

    m.def("Isotone_T", Isotone_T, "Isotone_T");
    m.def("compute_mse_loss", compute_mse_loss, "compute_mse_loss");
}
