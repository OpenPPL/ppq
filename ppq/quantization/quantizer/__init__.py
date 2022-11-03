from .base import BaseQuantizer
from .DSPQuantizer import PPL_DSP_Quantizer, PPL_DSP_TI_Quantizer
from .MetaxQuantizer import MetaxChannelwiseQuantizer, MetaxTensorwiseQuantizer
from .MyQuantizer import ExtQuantizer
from .NXPQuantizer import NXP_Quantizer
from .RKNNQuantizer import RKNN_PerChannelQuantizer, RKNN_PerTensorQuantizer
from .PPLQuantizer import PPLCUDAQuantizer
# from .TRTQuantizer import TensorRTQuantizer
from .FPGAQuantizer import FPGAQuantizer
from .NCNNQuantizer import NCNNQuantizer
from .OpenvinoQuantizer import OpenvinoQuantizer
from .TengineQuantizer import TengineQuantizer
from .FP8Quantizer import GraphCoreQuantizer, TensorRTQuantizer_FP8
from .TensorRTQuantizer import TensorRTQuantizer, TensorRTQuantizer_InputOnly